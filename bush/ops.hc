#include "bush/tensor.hc"
#include "bush/autograd.hc"


// Gradient tracking helpers
static U0 GradUpdateOneVar(Tensor *A, Tensor *C, F32 (*func)(F32, F32), OpType op_type, U0 (*backward_fn)(Tensor *)) {
    if (A->requires_grad) {
        C->requires_grad = TRUE; 
        C->op = op_type; 
        C->num_parents = 1; 
        C->parents = MAlloc(sizeof(Tensor *));
        C->parents[0] = A;
        C->backward_fn = backward_fn;
    }
}

static U0 GradUpdateTwoVars(Tensor *A, Tensor *B, Tensor *C, F32 (*func)(F32, F32), OpType op_type, U0 (*backward_fn)(Tensor *)) {
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = TRUE; 
        C->op = op_type; 
        C->num_parents = 2; 
        C->parents = MAlloc(2 * sizeof(Tensor *));
        C->parents[0] = A;
        C->parents[1] = B;
        C->backward_fn = backward_fn;
    }
}

// Tensor Functions
static U0 TensorEwise(Tensor *A, Tensor *B, Tensor *C, F32 (*func)(F32, F32), OpType op_type, U0 (*backward_fn)(Tensor *)) {
    I64 i; 

    if (!A || !B || !C) return;
    if (A->ndim != B->ndim || A->ndim != C->ndim) return;
    for (i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != C->shape[i]) return; 
    }

    for (i = 0; i < A->size; i++) {
        C->data[i] = func(A->data[i], B->data[i]);
    }

    GradUpdateTwoVars(A, B, C, func, op_type, backward_fn);
}

static F64 AddFunc(F64 a, F64 b) { return a + b; }
static F64 SubFunc(F64 a, F64 b) { return a - b; }
static F64 MulFunc(F64 a, F64 b) { return a * b; }

Tensor* TensorAdd(Tensor *A, Tensor *B) {
    I64 i; 
    I64 j; 

    if (!A || !B) return NULL; 

    Tensor *C = TensorCreate(A->shape, A->ndim); 
    if (!C) return NULL; 

    if (A->ndim == 2 && B->ndim == 1 && A->shape[1] == B->shape[0]) {
        Tensor *C = TensorCreate(A->shape, A->ndim);
        if (!C) return NULL;

        for (i = 0; i < A->shape[0]; i++) {
            for (j = 0; j < A->shape[1]; j++) {
                C->data[i * A->shape[1] + j] = A->data[i * A->shape[1] + j] + B->data[j];
            }
        }

        GradUpdateTwoVars(A, B, C, AddFunc, OP_ADD, BackwardAdd);
        return C; 
    }

    TensorEwise(A, B, C, AddFunc, OP_ADD, BackwardAdd);
    return C;
}

Tensor* TensorSub(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL; 

    Tensor *C = TensorCreate(A->shape, A->ndim); 
    if (!C) return NULL; 

    TensorEwise(A, B, C, SubFunc, OP_SUB, BackwardSub);
    return C;
}

Tensor* TensorEwiseMul(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL; 

    Tensor *C = TensorCreate(A->shape, A->ndim);
    if (!C) return NULL;

    TensorEwise(A, B, C, MulFunc, OP_MUL, BackwardMul);
    return C;
}

Tensor* TensorMatMul(Tensor *A, Tensor *B) {
    I64 i; 
    I64 j; 
    I64 k;

    if (!A || !B) return NULL;
    
    if (A->ndim == 1 && B->ndim == 1) {
        if (A->shape[0] != B->shape[0]) return NULL;

        Tensor *C = TensorCreate((size_t[]){1}, 1);
        if (!C) return NULL;
        
        F32 acc = 0.0f;
        for (i = 0; i < A->shape[0]; i++) {
            acc += A->data[i] * B->data[i];
        }
        C->data[0] = acc;

        GradUpdateTwoVars(A, B, C, NULL, OP_MATMUL, BackwardMatMul);
        return C;
    } else if (A->ndim == 2 && B->ndim == 1) {
        if (A->shape[1] != B->shape[0]) return NULL;

        Tensor *C = TensorCreate((size_t[]){A->shape[0]}, 1);
        if (!C) return NULL;

        for (i = 0; i < A->shape[0]; i++) {
            F32 acc = 0.0f;
            for (k = 0; k < A->shape[1]; k++) {
                acc += A->data[i * A->shape[1] + k] * B->data[k];
            }
            C->data[i] = acc;
        }

        GradUpdateTwoVars(A, B, C, NULL, OP_MATMUL, BackwardMatMul);
        return C;
    } else if (A->ndim == 1 && B->ndim == 2) {
        if (A->shape[0] != B->shape[0]) return NULL;

        Tensor *C = TensorCreate((size_t[]){B->shape[1]}, 1);
        if (!C) return NULL;

        for (j = 0; j < B->shape[1]; j++) {
            F32 acc = 0.0f;
            for (k = 0; k < A->shape[0]; k++) {
                acc += A->data[k] * B->data[k * B->shape[1] + j];
            }
            C->data[j] = acc;
        }

        GradUpdateTwoVars(A, B, C, NULL, OP_MATMUL, BackwardMatMul);
        return C;
    } else {
        return NULL; 
    }
}

Tensor* TensorTranspose(Tensor *A) {
    I64 i; 
    I64 j;

    if (!A) return NULL;
    if (A->ndim != 2) return NULL;

    Tensor *C = TensorCreate({A->shape[1], A->shape[0]}, 2);
    if (!C) return NULL;

    for (i = 0; i < A->shape[0]; i++) {
        for (j = 0; j < A->shape[1]; j++) {
            C->data[j * C->shape[1] + i] = A->data[i * A->shape[1] + j];
        }
    }

    GradUpdateOneVar(A, C, NULL, OP_TRANSPOSE, BackwardTranspose);

    return C;
}

// Activation Functions

