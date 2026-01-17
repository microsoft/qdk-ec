use crate::{BitMatrix, BitVec};
use pyo3::conversion::IntoPyObject;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use std::ffi::{CStr, CString};

const BITMATRIX_PTR_CAPSULE_NAME: &CStr = c"binar.BitMatrix.pointer";
const BITVEC_PTR_CAPSULE_NAME: &CStr = c"binar.BitVec.pointer";

/// A Send wrapper to satisfy `PyCapsule::new` requirements.
#[repr(transparent)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

impl<'py> IntoPyObject<'py> for BitMatrix {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // Box so we control the allocation.
        // The receiving side uses Box::from_raw to reclaim ownership.
        let raw_ptr = Box::into_raw(Box::new(self));
        let capsule = match PyCapsule::new(py, SendPtr(raw_ptr), None) {
            Ok(c) => c,
            Err(e) => {
                drop(unsafe { Box::from_raw(raw_ptr) });
                return Err(e);
            }
        };
        let binar = PyModule::import(py, intern!(py, "binar"))?;
        let class = binar.getattr(intern!(py, "BitMatrix"))?;
        class.call_method1(intern!(py, "_from_capsule"), (capsule,))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for &'a BitMatrix {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let capsule = ob.call_method0(intern!(ob.py(), "_as_capsule"))?;
        let capsule = capsule.cast::<PyCapsule>()?;
        let pointer = capsule.pointer_checked(Some(BITMATRIX_PTR_CAPSULE_NAME))?;

        // SAFETY: _as_capsule wraps &BitMatrix in SendPtr, and PyCapsule boxes it.
        // So pointer is *mut SendPtr<*mut BitMatrix> = *mut *mut BitMatrix (repr transparent).
        // We read through the indirection to get *const BitMatrix, borrowed for 'a.
        Ok(unsafe { &*pointer.cast::<*const BitMatrix>().read() })
    }
}

impl<'py> IntoPyObject<'py> for BitVec {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // Box so we control the allocation.
        // The receiving side uses Box::from_raw to reclaim ownership.
        let raw_ptr = Box::into_raw(Box::new(self));
        let capsule = match PyCapsule::new(py, SendPtr(raw_ptr), None) {
            Ok(c) => c,
            Err(e) => {
                drop(unsafe { Box::from_raw(raw_ptr) });
                return Err(e);
            }
        };
        let binar = PyModule::import(py, intern!(py, "binar"))?;
        let cls = binar.getattr(intern!(py, "BitVector"))?;
        cls.call_method1(intern!(py, "_from_capsule"), (capsule,))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for &'a BitVec {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let capsule = ob.call_method0(intern!(ob.py(), "_as_capsule"))?;
        let capsule = capsule.cast::<PyCapsule>()?;
        let pointer = capsule.pointer_checked(Some(BITVEC_PTR_CAPSULE_NAME))?;

        // SAFETY: _as_capsule wraps &BitVec in SendPtr, and PyCapsule boxes it.
        // So pointer is *mut SendPtr<*mut BitVec> = *mut *mut BitVec (repr transparent).
        // We read through the indirection to get *const BitVec, borrowed for 'a.
        Ok(unsafe { &*pointer.cast::<*const BitVec>().read() })
    }
}

#[must_use]
pub fn bitmatrix_from_capsule(capsule: &Bound<'_, PyCapsule>) -> Option<BitMatrix> {
    // PyCapsule::new boxes our SendPtr<*mut BitMatrix>, so capsule.pointer() returns
    // *mut SendPtr<*mut BitMatrix>, which is *mut *mut BitMatrix due to repr(transparent).
    // We read through both levels of indirection to reclaim the original Box.
    let ptr_ptr = capsule.pointer_checked(None).ok()?.cast::<*mut BitMatrix>();
    let ptr = unsafe { ptr_ptr.read() };
    Some(unsafe { *Box::from_raw(ptr) })
}

/// # Errors
/// Returns an error if the capsule cannot be created.
pub fn bitmatrix_as_capsule<'py>(matrix: &BitMatrix, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
    let pointer = SendPtr(std::ptr::from_ref(matrix).cast_mut());
    PyCapsule::new(py, pointer, Some(CString::from(BITMATRIX_PTR_CAPSULE_NAME)))
}

#[must_use]
pub fn bitvec_from_capsule(capsule: &Bound<'_, PyCapsule>) -> Option<BitVec> {
    // PyCapsule::new boxes our SendPtr<*mut BitVec>, so capsule.pointer() returns
    // *mut SendPtr<*mut BitVec>, which is *mut *mut BitVec due to repr(transparent).
    // We read through both levels of indirection to reclaim the original Box.
    let ptr_ptr = capsule.pointer_checked(None).ok()?.cast::<*mut BitVec>();
    let ptr = unsafe { ptr_ptr.read() };
    Some(unsafe { *Box::from_raw(ptr) })
}

/// # Errors
/// Returns an error if the capsule cannot be created.
pub fn bitvec_as_capsule<'py>(bitvec: &BitVec, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
    let pointer = SendPtr(std::ptr::from_ref(bitvec).cast_mut());
    PyCapsule::new(py, pointer, Some(CString::from(BITVEC_PTR_CAPSULE_NAME)))
}
