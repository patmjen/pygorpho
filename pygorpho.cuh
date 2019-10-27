#ifndef PYGORPHO_CUH__
#define PYGORPHO_CUH__

#include <stdexcept>
#include <cinttypes>

#ifdef _WIN32
#    define PYGORPHO_API __declspec(dllexport)
#else
#    define PYGORPHO_API
#endif

// Copied from: numpy/core/include/numpy/ndarraytypes.h
enum NPY_TYPES : int {
    NPY_BOOL = 0,
    NPY_BYTE, NPY_UBYTE,
    NPY_SHORT, NPY_USHORT,
    NPY_INT, NPY_UINT,
    NPY_LONG, NPY_ULONG,
    NPY_LONGLONG, NPY_ULONGLONG,
    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
    NPY_OBJECT = 17,
    NPY_STRING, NPY_UNICODE,
    NPY_VOID,
    /*
     * New 1.6 types appended, may be integrated
     * into the above in 2.0.
     */
    NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,

    NPY_NTYPES,
    NPY_NOTYPE,
    //NPY_CHAR NPY_ATTR_DEPRECATE("Use NPY_STRING"),
    NPY_USERDEF = 256,  /* leave room for characters */

    /* The number of types not including the new 1.6 types */
    NPY_NTYPES_ABI_COMPATIBLE = 21
};;

enum ErrorCode : int {
    SUCCESS = 0,
    ERR_BAD_MORPH_OP = 1,
    ERR_BAD_TYPE = 2,
    ERR_UNCAUGHT_EXCEPTION = 3
};

enum PyMorphOp : int {
    MOP_DILATE = 0,
    MOP_ERODE = 1
};

#define TRY_OR_RETURN_ERROR(expr) try {\
    expr \
} catch (int err) { \
    return err; \
} catch (...) { \
    return ERR_UNCAUGHT_EXCEPTION; \
}

#define typeDispatch(type, func, ...) [&]() \
{ \
    switch (type) { \
	case NPY_BOOL: return func<bool>(__VA_ARGS__); \
	case NPY_BYTE: return func<int8_t>(__VA_ARGS__); \
	case NPY_UBYTE: return func<uint8_t>(__VA_ARGS__); \
	case NPY_SHORT: return func<short>(__VA_ARGS__); \
	case NPY_USHORT: return func<unsigned short>(__VA_ARGS__); \
	case NPY_INT: return func<int>(__VA_ARGS__); \
	case NPY_UINT: return func<unsigned int>(__VA_ARGS__); \
	case NPY_LONG: return func<long>(__VA_ARGS__); \
	case NPY_ULONG: return func<unsigned long>(__VA_ARGS__); \
	case NPY_LONGLONG: return func<long long>(__VA_ARGS__); \
	case NPY_ULONGLONG: return func<unsigned long long>(__VA_ARGS__); \
	case NPY_FLOAT: return func<float>(__VA_ARGS__); \
	case NPY_DOUBLE: return func<double>(__VA_ARGS__); \
    default: throw ERR_BAD_TYPE; \
	} \
}()

#endif // PYGORPHO_CUH__