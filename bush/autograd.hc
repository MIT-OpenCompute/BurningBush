#include <bush/tensor.hc>

U0 *Calloc(U64 count, U64 size) {
    U64 total = count * size; 
    U0 *ptr = MAlloc(total); 
    if (ptr) MemSet(ptr, 0, total); 
    return ptr; 
}

U0 BackwardAdd(Tensor *C) {
    I64 i; 
    I64 j; 

    if (!C || !C->parents || C->num_parents < 2) return; 

    Tensor *A = C->parents[0]; 
    Tensor *B = C->parents[1]; 

    if (A->requires_grad) {
        if (!A->grad) A->grad = Calloc(A->size, sizeof(F32));
        for (i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i]; 
        }
    }

    if (B->requires_grad) {
        if (!B->grad) B->grad = Calloc(B->size, sizeof(F32));
        for (i = 0; i < A->shape[0]; i++) {
            for (j = 0; j < B->shape[0]; j++) {
                B->grad[j] += C->grad[i * A->shape[1] + j];
            }
        }
    } else {
        for (i = 0; i < B->size; i++) {
            B->grad[i] += C->grad[i];
        }
    }
}

U0 BackwardSub(Tensor *C) {
    I64 i; 
    I64 j; 

    if (!C || !C->parents || C->num_parents < 2) return; 

    Tensor *A = C->parents[0]; 
    Tensor *B = C->parents[1]; 

    if (A->requires_grad) {
        if (!A->grad) A->grad = Calloc(A->size, sizeof(F32));
        for (i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i]; 
        }
    }

    if (B->requires_grad) {
        if (!B->grad) B->grad = Calloc(B->size, sizeof(F32));
        for (i = 0; i < B->size; i++) {
            B->grad[i] -= C->grad[i];
        }
    }
}

U0 BackwardMul(Tensor *C) {
    I64 i; 

    if (!C || !C->parents || C->num_parents < 2) return; 

    Tensor *A = C->parents[0]; 
    Tensor *B = C->parents[1]; 

    if (A->requires_grad) {
        if (!A->grad) A->grad = Calloc(A->size, sizeof(F32)); 
        for (i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i] * B->data[i]; 
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = Calloc(B->size, sizeof(F32)); 
        for (i = 0; i < B->size; i++) {
            B->grad[i] += C->grad[i] * A->data[i]; 
        }
    }
}

U0 BackwardMatMul(Tensor *C) {
    I64 i; 
    I64 j; 

    if (!C || C->num_parents != 2) return;

    Tensor *A = C->parents[0]; 
    Tensor *B = C->parents[1]; 

    if (A->ndim == 1 && B->ndim == 1) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = Calloc(A->size, sizeof(F32)); 

        }

    } else if (A->ndim == 2 && B->ndim == 1) {

    } else if (A->ndim == 1 && B->ndim == 2) {

    } else if (A->ndim == 2 && B->ndim == 2) {

    }


}