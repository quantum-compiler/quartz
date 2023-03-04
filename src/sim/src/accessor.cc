#include "cuda_helper.h"

namespace sim {

using namespace Legion;

template <typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime) {
  AccessorRO<DT, dim> const acc(region, fid);
  rect = runtime->get_index_space_domain(ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
}

template <typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR() {}

GenericTensorAccessorR::GenericTensorAccessorR(DataType _data_type,
                                               Legion::Domain _domain,
                                               void const *_ptr)
    : data_type(_data_type), domain(_domain), ptr(_ptr) {}

GenericTensorAccessorR::GenericTensorAccessorR(
    GenericTensorAccessorW const &acc)
    : data_type(acc.data_type), domain(acc.domain), ptr(acc.ptr) {}

GenericTensorAccessorR::GenericTensorAccessorR()
    : data_type(DT_NONE), domain(Domain::NO_DOMAIN), ptr(nullptr) {}


float const *GenericTensorAccessorR::get_float_ptr() const {
  if (data_type == DT_FLOAT)
    return static_cast<float const *>(ptr);
  else {
    assert(false && "Invalid Accessor Type");
    return static_cast<float const *>(nullptr);
  }
}

double const *GenericTensorAccessorR::get_double_ptr() const {
  if (data_type == DT_DOUBLE)
    return static_cast<double const *>(ptr);
  else {
    assert(false && "Invalid Accessor Type");
    return static_cast<double const *>(nullptr);
  }
}

void const *GenericTensorAccessorR::get_void_ptr() const {
    return ptr;
}

template <typename DT, int dim>
TensorAccessorW<DT, dim>::TensorAccessorW(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime,
                                          bool readOutput) {
  rect = runtime->get_index_space_domain(ctx, req.region.get_index_space());
  if (readOutput) {
    AccessorRW<DT, dim> const acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
  } else {
    AccessorWO<DT, dim> const acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
    // FIXME: currently we zero init the region if not read output
    // assign_kernel<DT><<<GET_BLOCKS(rect.volume()), CUDA_NUM_THREADS>>>(
    //    ptr, rect.volume(), 0.0f);
    // checkCUDA(cudaDeviceSynchronize());
  }
}

template <typename DT, int dim>
TensorAccessorW<DT, dim>::TensorAccessorW() {}

GenericTensorAccessorW::GenericTensorAccessorW(DataType _data_type,
                                               Legion::Domain _domain,
                                               void *_ptr)
    : data_type(_data_type), domain(_domain), ptr(_ptr) {}

GenericTensorAccessorW::GenericTensorAccessorW()
    : data_type(DT_NONE), domain(Domain::NO_DOMAIN), ptr(nullptr) {}

float *GenericTensorAccessorW::get_float_ptr() const {
  if (data_type == DT_FLOAT)
    return static_cast<float *>(ptr);
  else {
    assert(false && "Invalid Accessor Type");
    return static_cast<float *>(nullptr);
  }
}

double *GenericTensorAccessorW::get_double_ptr() const {
  if (data_type == DT_DOUBLE)
    return static_cast<double *>(ptr);
  else {
    assert(false && "Invalid Accessor Type");
    return static_cast<double *>(nullptr);
  }
}

void *GenericTensorAccessorW::get_void_ptr() const {
    return ptr;
}

template <typename DT>
const DT *helperGetTensorPointerRO(PhysicalRegion region,
                                   RegionRequirement req,
                                   FieldID fid,
                                   Context ctx,
                                   Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorR<DT, DIM> acc(region, req, fid, ctx, runtime);              \
    return acc.ptr;                                                            \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

template <typename DT>
DT *helperGetTensorPointerRW(PhysicalRegion region,
                             RegionRequirement req,
                             FieldID fid,
                             Context ctx,
                             Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorW<DT, DIM> acc(                                              \
        region, req, fid, ctx, runtime, true /*readOutput*/);                  \
    return acc.ptr;                                                            \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

template <typename DT>
DT *helperGetTensorPointerWO(PhysicalRegion region,
                             RegionRequirement req,
                             FieldID fid,
                             Context ctx,
                             Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorW<DT, DIM> acc(                                              \
        region, req, fid, ctx, runtime, false /*readOutput*/);                 \
    return acc.ptr;                                                            \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default: {
      fprintf(stderr, "Unsupported accessor dimension");
      assert(false);
      return NULL;
    }
  }
}

GenericTensorAccessorR
    helperGetGenericTensorAccessorRO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  void const *ptr = nullptr;
  switch (datatype) {
    case DT_FLOAT: {
      ptr = helperGetTensorPointerRO<float>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_DOUBLE: {
      ptr = helperGetTensorPointerRO<double>(region, req, fid, ctx, runtime);
      break;
    }
    default: {
      assert(false);
    }
  }
  return GenericTensorAccessorR(datatype, domain, ptr);
}

GenericTensorAccessorW
    helperGetGenericTensorAccessorWO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  void *ptr = nullptr;
  switch (datatype) {
    case DT_FLOAT: {
      ptr = helperGetTensorPointerWO<float>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_DOUBLE: {
      ptr = helperGetTensorPointerWO<double>(region, req, fid, ctx, runtime);
      break;
    }
    default: {
      assert(false);
    }
  }
  return GenericTensorAccessorW(datatype, domain, ptr);
}

GenericTensorAccessorW
    helperGetGenericTensorAccessorRW(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime) {
  Domain domain =
      runtime->get_index_space_domain(ctx, req.region.get_index_space());
  void *ptr = nullptr;
  switch (datatype) {
    case DT_FLOAT: {
      ptr = helperGetTensorPointerRW<float>(region, req, fid, ctx, runtime);
      break;
    }
    case DT_DOUBLE: {
      ptr = helperGetTensorPointerRW<double>(region, req, fid, ctx, runtime);
      break;
    }
    default: {
      assert(false);
    }
  }
  return GenericTensorAccessorW(datatype, domain, ptr);
}

#define DIMFUNC(DIM)                                                           \
  template class TensorAccessorR<float, DIM>;                                  \
  template class TensorAccessorR<double, DIM>;                                \
  template class TensorAccessorW<float, DIM>;                                  \
  template class TensorAccessorW<double, DIM>;
LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC

template float const *helperGetTensorPointerRO(PhysicalRegion region,
                                               RegionRequirement req,
                                               FieldID fid,
                                               Context ctx,
                                               Runtime *runtime);
template float *helperGetTensorPointerRW(PhysicalRegion region,
                                         RegionRequirement req,
                                         FieldID fid,
                                         Context ctx,
                                         Runtime *runtime);
template float *helperGetTensorPointerWO(PhysicalRegion region,
                                         RegionRequirement req,
                                         FieldID fid,
                                         Context ctx,
                                         Runtime *runtime);

template double const *helperGetTensorPointerRO(PhysicalRegion region,
                                                RegionRequirement req,
                                                FieldID fid,
                                                Context ctx,
                                                Runtime *runtime);
template double *helperGetTensorPointerRW(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime);
template double *helperGetTensorPointerWO(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime *runtime);

}; // namespace sim
