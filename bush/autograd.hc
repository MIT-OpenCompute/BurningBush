#include "bush/tensor.hc"

static U0 *Calloc(U64 count, U64 size) {
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

U0 BackwardScale(Tensor *C) {
    I64 i; 

    if (!C || !C->parents || C->num_parents < 2) return; 

    Tensor *A = C->parents[0]; 
    Tensor *B = C->parents[1]; 

    if (A->requires_grad) {
        if (!A->grad) A->grad = Calloc(A->size, sizeof(F32)); 
        for (i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i] * B->data[0]; 
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = Calloc(B->size, sizeof(F32)); 
        F32 acc = 0.0f; 
        for (i = 0; i < B->size; i++) {
            acc += C->grad[i] * A->data[i]; 
        }
        B->grad[0] += acc; 
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
            for (i = 0; i < A->size; i++) {
                A->grad[i] += C->grad[0] * B->data[i]; 
            }
        }

        if (B->requires_grad) {
            if (!B->grad) B->grad = Calloc(B->size, sizeof(F32)); 
            for (i = 0; i < B->size; i++) {
                B->grad[i] += C->grad[0] * A->data[i]; 
            }
        }

    } else if (A->ndim == 2 && B->ndim == 1) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = Calloc(A->size, sizeof(F32)); 
            for (i = 0; i < A->shape[0]; i++) {
                for (j = 0; j < A->shape[1]; j++) {
                    A->grad[i * A->shape[1] + j] += C->grad[i] * B->grad[j];
                }
            }
        }

        if (B->requires_grad) {
            if (!B->grad) B->grad = Calloc(B->size, sizeof(F32)); 
            for (j = 0; j < B->shape[0]; j++) {
                float acc = 0.0f; 
                for (i = 0; i < A->shape[0]; i++) {
                    acc += A->data[i * A->shape[1] + j] * C->grad[i]; 
                }
                B->grad[j] += acc;
            }
        }

    } else if (A->ndim == 1 && B->ndim == 2) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = Calloc(A->size, sizeof(F32)); 
            for (i = 0; i < A->shape[0]; i++) {
                F32 acc = 0.0f; 
                for (j = 0; j < B->shape[1]; j++) {
                    acc += C->grad[j] * B->data[i * B->shape[1] + j]; 
                }
                A->grad[i] += acc; 
            }
        }

        if (B->requires_grad) {
            if (!B->grad) B->grad = Calloc(B->size, sizeof(F32)); 
            for (i = 0; i < B->shape[0]; i++) {
                for (j = 0; j < B->shape[1]; j++) {
                    B->grad[i * B->shape[1] + j] += A->data[i] * C->grad[j];
                }
            }
        }

    } else if (A->ndim == 2 && B->ndim == 2) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = Calloc(A->size, sizeof(F32)); 
            for (i = 0; i < A->shape[0]; i++) {
                for (j = 0; j < A->shape[1]; j++) {
                    F32 acc = 0.0f; 
                    for (k = 0; k < B->shape[1]; k++) {
                        acc += C->grad[i * B->shape[1] + k] * B->data[j * B->shape[1] + k];
                    }
                    A->grad[i * A->shape[1] + j] += acc;
                }
            }
        }

        if (B->requires_grad) {
            if (!B->grad) B->grad = Calloc(B->size, sizeof(F32)); 
            for (i = 0; i < B->shape[0]; i++) {
                for (j = 0; j < B->shape[1]; j++) {
                    F32 acc = 0.0f; 
                    for (k = 0; k < A->shape[1]; k++) {
                        acc += A->data[i * A->shape[1] + k] * C->grad[i * B->shape[1] + j];
                    }
                    B->grad[i * B->shape[1] + j] += acc;
                }
            }
        }
    }
}

U0 BackwardTranspose(Tensor *C) {
    I64 i; 
    I64 j; 

    if (!C || !C->parents || C->num_parents < 1) return;

    Tensor *A = C->parents[0]; 

    if (!A || A->ndim != 2) return; 

    if (A->requires_grad) {
        if (!A->grad) A->grad = Calloc(A->size, sizeof(F32)); 
        for (i = 0; i < A->shape[0]; i++) {
            for (j = 0; j < A->shape[1]; j++) {
                A->grad[i * A->shape[1] + j] += C->grad[j * A->shape[0] + i];
            }
        }
    }
}

// Activation function gradients
U0 BackwardRelu(Tensor *A) {
    I64 i;

    if (!A || !A->parents || A->num_parents < 1) return;

    Tensor *Z = A->parents[0]; 
    if (!Z) return; 

    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = Calloc(Z->size, sizeof(F32)); 
        for (i = 0; i < Z->size; i++) {
            Z->grad[i] += A->grad[i] * (Z->data[i] > 0 ? 1.0f : 0.0f); 
        }
    }
}

U0 BackwardSigmoid(Tensor *A) {
    I64 i; 

    Tensor *Z = A->parents[0]; 

    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = Calloc(Z->size, sizeof(F32)); 
        for (i = 0; i < Z->size; i++) {
            F32 sig = A->data[i]; 
            Z->grad[i] += A->grad[i] * sig * (1.0f - sig); 
        }
    }
}

U0 BackwardTanh(Tensor *A) {
    I64 i; 

    if (!A || !A->parents || A->num_parents < 1) return;

    Tensor *Z = A->parents[0]; 

    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = Calloc(Z->size, sizeof(F32)); 
        for (i = 0; i < Z->size; i++) {
            F32 t = A->data[i]; 
            Z->grad[i] += A->grad[i] * (1.0f - t * t); 
        }
    }
}

U0 BackwardSoftmax(Tensor *A) { 
    I64 i; 
    I64 b; 
    I64 j; 

    if (!A || !A->parents || A->num_parents < 1) return;

    Tensor *Z = A->parents[0]; 

    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = Calloc(Z->size, sizeof(F32)); 

        I64 batch_size = (Z->ndim == 2) ? Z->shape[0] : 1; 
        I64 num_classes = (Z->ndim == 2) ? Z->shape[1] : Z->size; 

        for (b = 0; b < batch_size; b++) {
            I64 offset = b * num_classes; 
            for (i = 0; i < num_classes; i++) {
                for (j = 0; j < num_classes; j++) {
                    F32 delta = (i == j) ? 1.0f : 0.0f; 
                    Z->grad[offset + i] += A->grad[offset + j] * A->data[offset + j] * (delta - A->data[offset + i]); 
                }
            }
        }
    }
}

// Loss function gradients
U0 BackwardMSE(Tensor *L) {
    I64 i; 

    Tensor *predicted = L->parents[0]; 
    Tensor *targets = L->parents[1]; 

    if (predictions->requires_grad) {
        if (!targets->grad) targets->grad = Calloc(targets->size, sizeof(F32)); 
        for (i = 0; i < targets->size; i++) {
            targets->grad[i] = (2.0f / targets->size) * (predictions->data[i] - targets->data[i]) * L->grad[0]; 
        }
    }

    if (targets->required_grad) {
        if (!targets->grad) targets->grad = Calloc(targets->size, sizeof(F32)); 
        for (i = 0; i < targets->size; i++) {
            targets->grad[i] -= (2.0f / targets->size) * (predictions->data[i] - targets->data[i]) * L->grad[0]; 
        }
    }
}

U0 BackwardCrossEntropy(Tensor *L) {
    I64 i; 

    Tensor *predictions = L->parents[0]; 
    Tensor *targets = L->parents[1]; 
    float epsilon = 1e-7f; 

    if (predictions->requires_grad) {
        if (!predictions->grad) predictions->grad = Calloc(predictions->size, sizeof(F32)); 
        for (i = 0; i < targets->size; i++) {
            F32 pred = predictions->data[i]; 
            pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred); 
            predictions->grad[i] += (-targets->data[i] / pred) * L->grad[0]; 
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) targets->grad = Calloc(targets->size, sizeof(F32));
        for (i = 0; i < targets->size; i++) {
            F32 pred = predictions->data[i];
            pred = pred < epsilon ? epsilon : pred; 
            targets->grad[i] -= (-logf(pred)) * L->grad[0]; 
        }
    }
}

U0 BackBinaryCrossEntropy (Tensor *L) {
    I64 i; 

    Tensor *predictions = L->parents[0]; 
    Tensor *targets = L->parents[1]; 
    F32 epsilon = 1e-7f; 

    if (predictions->requires_grad) {
        if (!predictions->grad) predictions->grad = Calloc(predictions->size, sizeof(F32));
        for (i = 0; i < predictions->size; i++) {
            F32 pred = predictions->data[i];
            pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred);
            predictions->grad[i] += (-(targets->data[i] / pred) + (1.0f - targets->data[i]) / (1.0f - pred)) * L->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) targets->grad = Calloc(targets->size, sizeof(F32));
        for (i = 0; i < targets->size; i++) {
            F32 pred = predictions->data[i];
            pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred);
            targets->grad[i] -= (-logf(pred) + -logf(1.0f - pred)) * L->grad[0];
        }
    }
}