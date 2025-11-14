enum TensorDType {
    TENSOR_DTYPE_F64 
    // God didn't intend for 32 bit floats apparently
}; 

enum TensorDevice {
    TENSOR_DEVICE_CPU
    // God did intend for multithreading but im rebelling
}; 

class TensorOp {
    I8 *name; 
    U0 (*Backward)(Tensor *self); 
    U0 (*FreeContext)(U0 *context); 
}

class Tensor {
    F64 *data; 
    F64 *grad; 

    I64 *shape; 
    I64 size; 
    I64 ndim; 

    U8 requires_grad;
    U8 is_leaf; 
    U8 is_view; 
    U8 owns_data;

    Tensor **parents;
    I32 n_parents;

    TensorDType dtype;
    TensorDevice device;
    TensorOp *op; 
    U0 *context; 

    /* ===== Tensor Creation ===== */

    Tensor *TensorEmpty(I64 *shape, I64 ndim) {

    }

    Tensor *TensorZeros(I64 *shape, I64 ndim) {

    }

    Tensor *TensorOnes(I64 *shape, I64 ndim) {

    }

    Tensor *TensorFull(I64 *shape, I64 ndim, F32 value) {

    }

    Tensor *TensorRandom(I64 *shape, I64 ndim, U32 seed) {

    }

    Tensor *TensorFromBuffer(F64 *buffer, I64 *shape, I64 ndim, U8 borrow_data) {

    }

    /* ===== Tensor Destruction ===== */

    U0 Free(Tensor *self) {

    }

    U0 Retain(Tensor *self) {

    }

    U0 Release(Tensor *self) {

    }

    /* ===== Tensor Shape Helpers ===== */

    I64 TensorNumel(Tensor *self) {

    }

    U0 TensorComputeStrides(Tensor *self) {

    }

    U0 TensorRecomputeSize(Tensor *self) {

    }

    Tensor *DeepCopy(Tensor *self) {

    }

    Tensor *ShallowCopy(Tensor *self) {

    }

    /* ===== Tensor Metadata Helpers ===== */

    Tensor *View(Tensor *self, I64 *new_shape, I64 ndim) {

    }

    Tensor *Transpose2D(Tensor *self) {

    }

    Tensor *Premute(Tensor *self, I64 *dims, I64 ndim) {

    }

    Tensor *SqueezeDim(Tensor *self, I64 dim) {

    }

    Tensor *UnsqueezeDim(Tensor *self, I64 dim) {

    }

    Tensor *SliceRaw(Tensor *self, I64 *start, I64 *end, I64 *step) {

    }

    /* ===== Tensor In-Place Helpers ===== */

    U0 Fill(Tensor *self, F64 value) {

    }

    U0 CopyInPlace(Tensor *self, Tensor *src) {

    }

    U0 ZeroData(Tensor *self) {

    }

    U0 ZeroGrad(Tensor *self) {

    }

    F32 GetIdx(Tensor *t, I64 *idx) {

    }

    F32 Get2D(Tensor *t, I64 row, I64 col) {

    }

    F32 SetIdx(Tensor *t, I64 *idx, F32 value) {

    }

    F32 Set2D(Tensor *t, I64 row, I64 col, F32 value) {

    }

    /* ===== Tensor Autograd Flags ===== */

    U0 SetRequiresGrad(Tensor *self, U8 requires_grad) {

    }

    U0 Detach(Tensor *self) {

    }

    Tensor *DetachedClone(Tensor *self) {

    }

    /* ===== Tensor Debug Utilities ===== */
    
    U0 PrintInfo(Tensor *self) {

    }

    U0 PrintData(Tensor *self) {

    }

    I32 Save(Tensor *t, I8 *path) {

    }

    Tensor *Load(I8 *path) {

    }
}
