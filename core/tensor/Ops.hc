/* ===== Elementwise ===== */

Tensor* TensorAdd(Tensor *a, Tensor *b) {

}

Tensor* TensorSub(Tensor *a, Tensor *b) {

}

Tensor* TensorMul(Tensor *a, Tensor *b) {

}

Tensor* TensorDiv(Tensor *a, Tensor *b) {

}

Tensor *TensorAddScalar(Tensor *a, F64 b) {

}

Tensor *TensorSubScalar(Tensor *a, F64 b) {

}

*Tensor *TensorMulScalar(Tensor *a, F64 b) {

}

Tensor *TensorDivScalar(Tensor *a, F64 b) {

}

Tensor *TensorNeg(Tensor *a) {

}

Tensor *TensorEwiseInv(Tensor *a) {

}

Tensor *TensorEwisePow(Tensor *a, F64 b) {

}

/* ===== Activations ===== */

Tensor *TensorRelu(Tensor *a) {

}

Tensor *TensorLeakyRelu(Tensor *a, F64 slope) {

}

Tensor *TensorSigmoid(Tensor *a) {

}

Tensor *TensorTanh(Tensor *a) {

}

Tensor *TensorGelu(Tensor *a) {

}

Tensor *TensorSoftplus(Tensor *a) {

}

Tensor *TensorSwish(Tensor *a) {

}

Tensor *TensorExp(Tensor *a) {

}

Tensor *TensorLog(Tensor *a) {

}

Tensor *TensorSqrt(Tensor *a) {

}

Tensor *TensorAbs(Tensor *a) {

}

Tensor *TensorClamp(Tensor *a, F64 min, F64 max) {

}

/* ===== Comparisons ===== */

Tensor *TensorGreater(Tensor *a, Tensor *b) {

}

Tensor *TensorLess(Tensor *a, Tensor *b) {

}

Tensor *TensorGreaterEqual(Tensor *a, Tensor *b) {

}

Tensor *TensorLessEqual(Tensor *a, Tensor *b) {

}

Tensor *TensorEqual(Tensor *a, Tensor *b) {

}

Tensor *TensorNotEqual(Tensor *a, Tensor *b) {

}

Tensor *TensorWhere(Tensor *condition, Tensor *a, Tensor *b) {

}

/* ===== Shape ===== */

Tensor *TensorReshape(Tensor *self, I64 *new_shape, I64 ndim) {

}

Tensor *TensorFlatten(Tensor *self) {

}

Tensor *TensorView(Tensor *self, I64 *new_shape, I64 ndim) {

}

Tensor *TensorTranspose(Tensor *self) {

}

Tensor *TensorPermute(Tensor *self, I64 *dims, I64 ndim) {

}

Tensor *TensorSqueeze(Tensor *self, I64 dim) {

}

Tensor *TensorUnsqueeze(Tensor *self, I64 dim) {

}

Tensor *TensorConcat(Tensor **tensors, I64 n, I64 dim) {

}

Tensor *TensorStack(Tensor **tensors, I64 n, I64 dim) {

}

Tensor *TensorSlice(Tensor *self, I64 *start, I64 *end, I64 *step) {

}

Tensor *TensorIndex(Tensor *self, I64 **indices, I64 *n_indices, I64 ndim) {

}

Tensor *TensorGather(Tensor *self, Tensor *indices, I64 dim) {

}

Tensor *TensorScatterAdd(Tensor *self, Tensor *indices, Tensor *src, I64 dim) {

}

/* ===== Reductions ===== */

Tensor *TensorSum(Tensor *a, I64 dim, U8 keepdim) {

}

Tensor *TensorMean(Tensor *a, I64 dim, U8 keepdim) {

}

Tensor *TensorMax(Tensor *a, I64 dim, U8 keepdim) {

}

Tensor *TensorMin(Tensor *a, I64 dim, U8 keepdim) {

}

Tensor *TensorVar(Tensor *a, I64 dim, U8 unbiased, U8 keepdim) {

}

Tensor *TensorStd(Tensor *a, I64 dim, U8 unbiased, U8 keepdim) {

}

Tensor *TensorL2Norm(Tensor *a, I64 dim, U8 keepdim) {

}

Tensor *TensorL1Norm(Tensor *a, I64 dim, U8 keepdim) {

}

/* ===== Linear Algebra ===== */

Tensor *TensorDot(Tensor *a, Tensor *b) {

}

Tensor *TensorMVMul(Tensor *a, Tensor *b) {

}

Tensor *TensorVMMul(Tensor *a, Tensor *b) {

}

Tensor *TensorBatchedMul(Tensor *a, Tensor *b) {

}

Tensor *TensorMatMul(Tensor *a, Tensor *b) {

}

Tensor *TensorLinear(Tensor *x, Tensor *w, Tensor *b) {

}

/* ===== Classifiers ===== */

Tensor *TensorSoftmax(Tensor *z, I64 dim) {

}

Tensor *TensorLogSoftmax(Tensor *z, I64 dim) {

}

Tensor *TensorLogSumExp(Tensor *z, I64 dim) {

}

/* ===== Loss Functions ===== */

Tensor *TensorMSELoss(Tensor *pred, Tensor *target, U8 reduction) {

}

Tensor *TensorL1Loss(Tensor *pred, Tensor *target, U8 reduction) {

}

Tensor *TensorCrossEntropyLoss(Tensor *pred, Tensor *target, U8 reduction) {

}

Tensor *TensorBinaryCrossEntropyLoss(Tensor *pred, Tensor *target, U8 reduction) {

}

Tensor *TensorBCELogitsLoss(Tensor *z, Tensor *target, U8 reduction) {

}

Tensor *TensorSmoothL1Loss(Tensor *pred, Tensor *target, F64 beta, U8 reduction) {

}

Tensor *TensorKLDivergence(Tensor *p, Tensor *q, U8 reduction) {

}

/* ===== Regularization ===== */

Tensor *TensorDropout(Tensor *a, F64 p, U8 training) {

}

Tensor *TensorAddGaussianNoise(Tensor *a, F64 stddev, U32 seed) {

}

/* ===== Utilities ===== */

Tensor *TensorCast(Tensor *a, TensorDType new_dtype) {

}

Tensor *TensorNormalizeL2(Tensor *a, I64 dim, F64 eps) {

}

Tensor *TensorBatchMean(Tensor *a, I64 batch_dim) {

}

Tensor *OpBatchStd(Tensor *a, I64 batch_dim, F64 eps) {

}