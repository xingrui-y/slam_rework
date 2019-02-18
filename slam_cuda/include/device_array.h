#ifndef __DEVICE_ARRAY__
#define __DEVICE_ARRAY__

#include <vector>
#include <atomic>
#include "utils.h"

template <class T>
struct PtrSz
{
	__device__ inline T &operator[](int x) const;
	__device__ inline operator T *() const;
	T *data;
	size_t size;
};

template <class T>
struct PtrStep
{
	__device__ inline T *ptr(int y = 0) const;
	T *data;
	size_t step;
};

template <class T>
struct PtrStepSz
{
	__device__ inline T *ptr(int y = 0) const;
	T *data;
	int cols, rows;
	size_t step;
};

template <class T>
struct DeviceArray
{

	DeviceArray();
	~DeviceArray();
	DeviceArray(size_t size_);
	DeviceArray(const std::vector<T> &vec);

	void create(size_t size_);

	void upload(const void *data_);
	void upload(const std::vector<T> &vec);
	void upload(const void *data_, size_t size_);

	void download(void *data_) const;
	void download(std::vector<T> &vec) const;
	void download(void *data_, size_t size_) const;

	void clear();
	void release();
	void copyTo(DeviceArray<T> &other) const;
	DeviceArray<T> &operator=(const DeviceArray<T> &other);
	operator T *() const;
	operator PtrSz<T>() const;

	void *data;
	size_t size;
	std::atomic<int> *ref;
};

template <class T>
struct DeviceArray2D
{
	DeviceArray2D();
	DeviceArray2D(int cols_, int rows_);
	~DeviceArray2D();

	void create(int cols_, int rows_);
	void upload(const void *data_);
	void upload(const void *data_, size_t step_);
	void upload(const void *data_, size_t step_, int cols_, int rows_);
	void download(void *data_, size_t step_) const;

	void clear();
	void release();
	void copyTo(DeviceArray2D<T> &other) const;
	DeviceArray2D<T> &operator=(const DeviceArray2D<T> &other);
	operator T *() const;
	operator PtrStep<T>() const;
	operator PtrStepSz<T>() const;

	void *data;
	size_t step;
	int cols, rows;
	std::atomic<int> *ref;
};

//------------------------------------------------------------------
// PtrSz
//------------------------------------------------------------------
template <class T>
__device__ inline T &PtrSz<T>::operator[](int x) const
{
	return data[x];
}

template <class T>
__device__ inline PtrSz<T>::operator T *() const
{
	return data;
}

//------------------------------------------------------------------
// PtrStep
//------------------------------------------------------------------
template <class T>
__device__ inline T *PtrStep<T>::ptr(int y) const
{
	return (T *)((char *)data + y * step);
}

//------------------------------------------------------------------
// PtrStepSz
//------------------------------------------------------------------
template <class T>
__device__ inline T *PtrStepSz<T>::ptr(int y) const
{
	return (T *)((char *)data + y * step);
}

//------------------------------------------------------------------
// DeviceArray
//------------------------------------------------------------------
template <class T>
DeviceArray<T>::DeviceArray() : data(0), ref(0), size(0)
{
}

template <class T>
DeviceArray<T>::DeviceArray(size_t size_) : data(0), ref(0), size(size_)
{
	create(size_);
}

template <class T>
DeviceArray<T>::DeviceArray(const std::vector<T> &vec) : data(0), ref(0), size(vec.size())
{
	create(size);
	upload(vec);
}

template <class T>
DeviceArray<T>::~DeviceArray()
{
	release();
}

template <class T>
void DeviceArray<T>::create(size_t size_)
{
	if (data)
		release();
	safe_call(cudaMalloc(&data, sizeof(T) * size_));
	size = size_;
	ref = new std::atomic<int>(1);
}

template <class T>
void DeviceArray<T>::upload(const void *data_)
{
	upload(data_, size);
}

template <class T>
void DeviceArray<T>::upload(const std::vector<T> &vec)
{
	upload(vec.data(), vec.size());
}

template <class T>
void DeviceArray<T>::upload(const void *data_, size_t size_)
{
	if (size_ > size)
		return;
	safe_call(cudaMemcpy(data, data_, sizeof(T) * size_, cudaMemcpyHostToDevice));
}

template <class T>
void DeviceArray<T>::download(void *data_) const
{
	download(data_, size);
}

template <class T>
void DeviceArray<T>::download(std::vector<T> &vec) const
{
	if (vec.size() != size)
		vec.resize(size);
	download((void *)vec.data(), vec.size());
}

template <class T>
void DeviceArray<T>::download(void *data_, size_t size_) const
{
	safe_call(cudaMemcpy(data_, data, sizeof(T) * size_, cudaMemcpyDeviceToHost));
}

template <class T>
void DeviceArray<T>::clear()
{
	safe_call(cudaMemset(data, 0, sizeof(T) * size));
}

template <class T>
void DeviceArray<T>::release()
{
	if (ref && ref->fetch_sub(1) == 1)
	{
		delete ref;
		if (data)
		{
			safe_call(cudaFree(data));
		}
	}

	size = 0;
	data = 0;
	ref = 0;
}

template <class T>
void DeviceArray<T>::copyTo(DeviceArray<T> &other) const
{
	if (!data)
	{
		other.release();
		return;
	}

	other.create(size);
	safe_call(cudaMemcpy(other.data, data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
}

template <class T>
DeviceArray<T> &DeviceArray<T>::operator=(const DeviceArray<T> &other)
{
	if (this != &other)
	{
		if (other.ref)
			other.ref->fetch_add(1);

		release();

		ref = other.ref;
		size = other.size;
		data = other.data;
	}

	return *this;
}

template <class T>
DeviceArray<T>::operator T *() const
{
	return (T *)data;
}

template <class T>
DeviceArray<T>::operator PtrSz<T>() const
{
	PtrSz<T> ps;
	ps.data = (T *)data;
	ps.size = size;
	return ps;
}

//------------------------------------------------------------------
// DeviceArray2D
//------------------------------------------------------------------
template <class T>
DeviceArray2D<T>::DeviceArray2D() : data(0), ref(0), step(0), cols(0), rows(0)
{
}

template <class T>
DeviceArray2D<T>::DeviceArray2D(int cols_, int rows_) : data(0), ref(0), step(0), cols(cols_), rows(rows_)
{
	create(cols_, rows_);
}

template <class T>
DeviceArray2D<T>::~DeviceArray2D()
{
	release();
}

template <class T>
void DeviceArray2D<T>::create(int cols_, int rows_)
{
	if (cols_ > 0 && rows_ > 0)
	{
		if (data)
			release();
		safe_call(cudaMallocPitch(&data, &step, sizeof(T) * cols_, rows_));

		cols = cols_;
		rows = rows_;
		ref = new std::atomic<int>(1);
	}
}

template <class T>
void DeviceArray2D<T>::upload(const void *data_)
{
	upload(data_, sizeof(T) * cols, cols, rows);
}

template <class T>
void DeviceArray2D<T>::upload(const void *data_, size_t step_)
{
	upload(data_, step_, cols, rows);
}

template <class T>
void DeviceArray2D<T>::upload(const void *data_, size_t step_, int cols_, int rows_)
{
	if (!data)
		create(cols_, rows_);

	safe_call(cudaMemcpy2D(data, step, data_, step_, sizeof(T) * cols_, rows_, cudaMemcpyHostToDevice));
}

template <class T>
void DeviceArray2D<T>::clear()
{
	safe_call(cudaMemset2D(data, step, 0, sizeof(T) * cols, rows));
}

template <class T>
void DeviceArray2D<T>::download(void *data_, size_t step_) const
{
	if (!data)
		return;
	safe_call(cudaMemcpy2D(data_, step_, data, step, sizeof(T) * cols, rows, cudaMemcpyDeviceToHost));
}

template <class T>
void DeviceArray2D<T>::release()
{
	if (ref && ref->fetch_sub(1) == 1)
	{
		delete ref;
		if (data)
			safe_call(cudaFree(data));
	}
	cols = rows = step = 0;
	data = ref = 0;
}

template <class T>
void DeviceArray2D<T>::copyTo(DeviceArray2D<T> &other) const
{
	if (!data)
		other.release();
	other.create(cols, rows);
	safe_call(cudaMemcpy2D(other.data, other.step, data, step, sizeof(T) * cols, rows, cudaMemcpyDeviceToDevice));
}

template <class T>
DeviceArray2D<T> &DeviceArray2D<T>::operator=(const DeviceArray2D<T> &other)
{
	if (this != &other)
	{
		if (other.ref)
			other.ref->fetch_add(1);
		release();

		data = other.data;
		step = other.step;
		cols = other.cols;
		rows = other.rows;
		ref = other.ref;
	}

	return *this;
}

template <class T>
DeviceArray2D<T>::operator T *() const
{
	return (T *)data;
}

template <class T>
DeviceArray2D<T>::operator PtrStep<T>() const
{
	PtrStep<T> ps;
	ps.data = (T *)data;
	ps.step = step;
	return ps;
}

template <class T>
DeviceArray2D<T>::operator PtrStepSz<T>() const
{
	PtrStepSz<T> psz;
	psz.data = (T *)data;
	psz.cols = cols;
	psz.rows = rows;
	psz.step = step;
	return psz;
}

#endif
