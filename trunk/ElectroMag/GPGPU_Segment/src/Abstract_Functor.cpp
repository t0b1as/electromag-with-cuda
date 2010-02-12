#include "Abstract Functor.h"
#include "X-Compat/Threading.h"
#include <cstdio>

AbstractFunctor::AbstractFunctor()
{
	// Initialize the mutex
	Threads::CreateMutex(&this->hRemapMutex);
}

AbstractFunctor::~AbstractFunctor()
{
	// Tidy up
	Threads::DestroyMutex(this->hRemapMutex);
}


unsigned long AbstractFunctor::AsyncFunctor(AbstractFunctor::AsyncParameters *parameters)
{
	AbstractFunctor *pObject = parameters->functorClass;
	Threads::MutexHandle *phMutex = &pObject->hRemapMutex;
	size_t functorID = parameters->functorIndex;
	size_t deviceID = functorID;
	unsigned long retVal;
	bool fail = true;
	bool reIterate = true;
	while(reIterate)
	{
		// In order for the syncronization mechanism to work, all 'functorClass' in a Run() call must point to the same object
		retVal = pObject->MainFunctor(functorID, deviceID);

		// Check to see whether functor completed without errors
		fail = pObject->FailOnFunctor(functorID);

		Threads::LockMutex(*phMutex);
		if(fail)
		{
			// The functor has failed; it needs to be remapped to another device
			if(pObject->nIdle)
			{
				// Idle device available; we can remap immediately

				// Get the last idle functor in the list, and remove it from the list
				deviceID = pObject->idleDevices[--pObject->nIdle];
				reIterate = true;
			}
			else
			{
				// No functor is idle, we need to add this to the the failed queue and leave
				pObject->failedFunctors[pObject->nFailed++] = functorID;
				reIterate = false;
			}
		}
		else
		{
			// the functor has succeded, this device can execute another functor
			if(pObject->nFailed)
			{
				// A Failed functor is available for processing
				functorID = pObject->failedFunctors[--pObject->nFailed];
				reIterate = true;
			}
			else
			{
				// No failed functors, we can add this to the idle queue and leave
				pObject->idleDevices[pObject->nIdle++] = deviceID;
				reIterate = false;
			}
		}
		Threads::UnlockMutex(*phMutex);
		
	}

	return retVal;
}

unsigned long AbstractFunctor::AsyncAuxFunctor(AbstractFunctor::AsyncParameters *parameters)
{
	return parameters->functorClass->AuxFunctor();
}

unsigned long AbstractFunctor::Run()
{
	// Allocate needed resources on each device
	this->AllocateResources();
	if(this->Fail()) return (1<<16);

	size_t nFunctors;
	// Create parameters for functors
	this->GenerateParameterList(&nFunctors);
	if(this->Fail()) return (2<<16);

	// Alocate resources for calling the async functors
	AbstractFunctor::AsyncParameters *launchParams = new AbstractFunctor::AsyncParameters[nFunctors];
	Threads::ThreadHandle * handles = new Threads::ThreadHandle[nFunctors];
	// Allocate and initialize resources that the async functors will use for syncronization
	this->idleDevices = new size_t[nFunctors];
	this->failedFunctors = new size_t[nFunctors];
	this->nFailed = this->nIdle = 0;

	for(size_t i = 0; i < nFunctors; i++)
	{
		unsigned long threadID;

		launchParams[i].functorClass = this;
		launchParams[i].functorIndex = i;
		launchParams[i].nFunctors = nFunctors;
		Threads::CreateNewThread((unsigned long (*)(void *))AbstractFunctor::AsyncFunctor, (void*)&launchParams[i], &handles[i], &threadID);

		// Set the name for the thread
		char threadName[512];
		sprintf(threadName, "AbstractFunctor Device %u", i);
		Threads::SetThreadName(threadID, threadName);
	}

	// Create thread for auxiliary functor
	unsigned long threadID;
	AbstractFunctor::AsyncParameters auxParams = {this, 0, 0};
	Threads::ThreadHandle hAuxFunctor;
	Threads::CreateNewThread((unsigned long (*)(void *))AbstractFunctor::AsyncAuxFunctor, &auxParams, &hAuxFunctor, &threadID);
	// Set the name for the thread
	Threads::SetThreadName(threadID, "Aux Functor");

	// Now wait for all functors to complete
	for(size_t i = 0; i < nFunctors; i++)
	{
		Threads::WaitForThread(handles[i]);
	}

	// Release resources used for syncronization
	delete [] this->idleDevices;
	delete [] this->failedFunctors;

	// Now terminate the auxiliary functor
	Threads::KillThread(hAuxFunctor);

	PostRun();

	return this->nFailed;
}
