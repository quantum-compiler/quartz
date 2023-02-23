#ifndef _CUDA_HELPER_H_
#define _CUDA_HELPER_H_
#include "legion.h"
#include "simulator_const.h"

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

class GenericTensorAccessorW {
public:
  GenericTensorAccessorW();
  GenericTensorAccessorW(DataType data_type, Legion::Domain domain, void *ptr);
  void *get_void_ptr() const;
  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
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
  int32_t const *get_int32_ptr() const;
  int64_t const *get_int64_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  DataType data_type;
  Legion::Domain domain;
  void const *ptr;
};

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
