enum OpType {
    OP_NONE, 
    OP_ADD, 
    OP_SUB, 
    OP_MUL, 
    OP_MATMUL, 
    OP_TRANSPOSE, 
    OP_RELU, 
    OP_SIGMOID, 
    OP_TANH, 
    OP_SOFTMAX,
    OP_MSE, 
    OP_CROSS_ENTROPY, 
    OP_BINARY_CROSS_ENTROPY
};

class Tensor {
    F32 *data; 
    F32 *grad; 
    I64 *shape; 
    I64 ndim; 
    I64 size; 

    Bool requires_grad;
    Bool owns_data; 
    OpType op_type;
    Tensor **parents;
    I64 num_parents;
    U0 (*backward_fn)(Tensor *self); 
    U8 *extra_data; 
};

// Autograd helper functions
U0 TopologicalSortUtil(Tensor *T, Tensor **visited, I64 visited_count, Tensor **stack, I64 *visited_count, Tensor **stack, I64 *stack_count, I64 max_size) {
    I64 i; 

    for (i = 0; i < *visited_count; i++) {
        if (visited[i] == T) return;
    }

    visited[(*visited_count)++] = T; 

    if (T->inputs) {
        for (i = 0; i < T->num_inputs; i++) {
            if (T->inputs[i] && T->inputs[i]->requires_grad) {
                TopologicalSortUtil(T->inputs[i], visited, visited_count, stack, stack_count, max_size);
            }
        }
    }

    if (*stack_count < max_size) {
        stack[(*stack_count)++] = T; 
    }
}

// Tensor creation/destruction
Tensor* TensorCreate(I64 *shape, I64 ndim) {
    I64 i;

    Tensor *T = MAlloc(sizeof(Tensor));
    if (!T) return NULL; 

    T->ndim = ndim; 
    T->shape = (I64 *)MAlloc(ndim * sizeof(I64)); 
    if (!T->shape) {
        Free(T); 
        return NULL; 
    }

    T->size = 1; 
    for (i = 0; i < ndim; i++) {
        T->shape[i] = shape[i]; 
        T->size *= shape[i];
    }

    T->data = MAlloc(T->size * sizeof(F32)); 
    if (!T->data) {
        Free(T->shape); 
        Free(T); 
        return NULL; 
    }

    T->grad = NULL; 
    T->requires_grad = FALSE;
    T->owns_data = TRUE;
    T->op_type = OP_NONE;
    T->parents = NULL;
    T->num_parents = 0;
    T->backward_fn = NULL;
    T->extra_data = NULL;
    return T;
}

Tensor* TensorCreate(I64 *shape, I64 ndim) {
    Tensor *T = TensorCreate(shape, ndim); 
    if (!T) return NULL; 

    MemSet(T->data, 0, T->size * sizeof(F32)); 
    return T; 
}

Tesnor* TensorOnes(I64 *shape, I64 ndim) {
    Tensor *T = TensorCreate(shape, ndim); 
    if (!T) return NULL; 

    I64 i; 
    for (i = 0; i < T->size; i++) {
        T->data[i] = 1.0f;
    }
    return T; 
}

Tensor* TensorRandn(I64 *shape, I64 ndim, I64 seed) {
    I64 i; 
    if (!T) return NULL; 

    Seed(seed);
    for (i = 0; i < T->size; i++) {
        F32 u1 = RANDF32();
        F32 u2 = RANDF32();
        F32 z0 = Sqrt(-2.0f * Log(u1)) * Cos(2.0f * M_PI * u2);
        T->data[i] = z0;
    }
    return T;
}

U0 TensorFree(Tensor *T) {
    if (!T) return; 

    if (T->owns_data) {
        if (T->data) Free(T->data);
        if (T->grad) Free(T->grad);
    }

    if (T->shape) Free(T->shape);
    if (T->parents) Free(T->parents);
    if (T->extra_data) Free(T->extra_data);

    Free(T); 
}

// Autograd functions
U0 TensorSetRequiresGrad(Tensor *T, Bool requires_grad) {
    if (!T) return; 
    if (T->grad) {
        MemSet(T->grad, 0, T->size * sizeof(F32));
    }
}

U0 TensorBackward(Tensor *T) {
    I64 i; 
    if (!T || !T->requires_grad) return; 

    if (!T->grad) {
        T->grad = MAlloc(T->size * sizeof(F32));
        for (i = 0; i < T->size; i++) {
            T->grad[i] = 0.0f;
        }
    }

    I64 max_size = 1024; 
    Tensor **visited = MAlloc(max_size * sizeof(Tensor *)); 
    Tensor **stack = MAlloc(max_size * sizeof(Tensor *));
    I64 visited_count = 0;
    I64 stack_count = 0;

    TopologicalSortUtil(T, visited, &visited_count, stack, &stack_count, max_size); 

    for (i = stack_count; i > 0; i--) {
        Tensor *current = stack[i - 1]; 
        if (current->backward_fn) {
            current->backward_fn(current);
        }
    }

    Free(visited); 
    Free(stack); 
}

U0 TensorZeroGrad(Tensor *T) {
    if (!T || !T->grad) return; 
    MemSet(T->grad, 0, T->size * sizeof(F32)); 
}

U0 TensorFill(Tensor *T, F32 value) {
    if (!T) return; 
    I64 i; 
    for (i = 0; i < T->size; i++) {
        T->data[i] = value;
    }
}

U0 TensorPrint(Tensor *T) {
    I64 i;
    I64 j;

    if (!T) return; 

    Print("%zu", T->shape[i]); 
    if (i < T->ndim - 1) {
        Print(" x ");
    }

    printf("], ndim=%zu, data=\n[", T->ndim);
    if (T->ndim == 2) {
        for (i = 0; i < T->shape[0]; i++) {
            Print("[");
            for (j = 0; j < T->shape[1]; j++) {
                Print("%f", T->data[i * T->shape[1] + j]);
                if (j < T->shape[1] - 1) {
                    Print(", ");
                }
            }
            Print("]");
            if (i < T->shape[0] - 1) {
                Print(",\n");
            }
        }
    } else {
        for (i = 0; i < T->size; i++) {
            Print("%.4f", T->data[i]); 
            if (i < T->size - 1) {
                Print(", "); 
            }
        }
    }
    Print("])\n"); 
}

Tensor* TensorCopy(Tensor *T) {
    if (!T) return NULL; 

    Tensor *C = TensorCreate(T->shape, T->ndim); 
    if (!C) return NULL; 

    MemCpy(C->data, T->data, T->size * sizeof(F32)); 

    C->grad = NULL;
    C->requires_grad = 0; 
    C->owns_data = TRUE; 
    C->op_type = OP_NONE;
    C->parents = NULL;
    C->num_parents = 0;
    C->backward_fn = NULL;
    C->extra_data = NULL;

    return C; 
}