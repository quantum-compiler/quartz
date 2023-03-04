#ifndef _CUDA_HELPER_H_
#define _CUDA_HELPER_H_
#include "legion.h"
#include "distributed_simulator_const.h"

namespace sim {

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS) {                                    \
      printf("Error: %s in line %d\n", custatevecGetErrorString(err),          \
             __LINE__);                                                        \
      return err;                                                              \
    }                                                                          \
  };

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__);     \
      return err;                                                              \
    }                                                                          \
  };

template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRO =
    Legion::FieldAccessor<READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorRW = Legion::
    FieldAccessor<READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
template <typename FT, int N, typename T = Legion::coord_t>
using AccessorWO = Legion::
    FieldAccessor<WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;
    
template <typename DT, int dim>
struct TensorAccessorR {
  TensorAccessorR(Legion::PhysicalRegion region,
                  Legion::RegionRequirement req,
                  Legion::FieldID fid,
                  Legion::Context ctx,
                  Legion::Runtime *runtime);
  TensorAccessorR();
  Legion::Rect<dim> rect;
  Legion::Memory memory;
  const DT *ptr;
};

template <typename DT, int dim>
struct TensorAccessorW {
  TensorAccessorW(Legion::PhysicalRegion region,
                  Legion::RegionRequirement req,
                  Legion::FieldID fid,
                  Legion::Context ctx,
                  Legion::Runtime *runtime,
                  bool readOutput = false);
  TensorAccessorW();
  Legion::Rect<dim> rect;
  Legion::Memory memory;
  DT *ptr;
};

class GenericTensorAccessorW {
public:
  GenericTensorAccessorW();
  GenericTensorAccessorW(DataType data_type, Legion::Domain domain, void *ptr);
  void *get_void_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  DataType data_type;
  Legion::Domain domain;
  void *ptr;
};

class GenericTensorAccessorR {
public:
  GenericTensorAccessorR();
  GenericTensorAccessorR(DataType data_type,
                         Legion::Domain domain,
                         void const *ptr);
  GenericTensorAccessorR(GenericTensorAccessorW const &acc);
  // GenericTensorAccessorR &operator=(GenericTensorAccessorW const &acc);
  void const *get_void_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  DataType data_type;
  Legion::Domain domain;
  void const *ptr;
};

template <typename DT>
const DT *helperGetTensorPointerRO(Legion::PhysicalRegion region,
                                   Legion::RegionRequirement req,
                                   Legion::FieldID fid,
                                   Legion::Context ctx,
                                   Legion::Runtime *runtime);

template <typename DT>
DT *helperGetTensorPointerWO(Legion::PhysicalRegion region,
                             Legion::RegionRequirement req,
                             Legion::FieldID fid,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

template <typename DT>
DT *helperGetTensorPointerRW(Legion::PhysicalRegion region,
                             Legion::RegionRequirement req,
                             Legion::FieldID fid,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

GenericTensorAccessorR
    helperGetGenericTensorAccessorRO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime);

GenericTensorAccessorW
    helperGetGenericTensorAccessorWO(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime);

GenericTensorAccessorW
    helperGetGenericTensorAccessorRW(DataType datatype,
                                     Legion::PhysicalRegion region,
                                     Legion::RegionRequirement req,
                                     Legion::FieldID fid,
                                     Legion::Context ctx,
                                     Legion::Runtime *runtime);
}; // namespace
#endif // _CUDA_HELPER_H_
