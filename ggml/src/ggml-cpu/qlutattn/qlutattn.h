#pragma once

#include "ggml-backend.h"
#include "ggml.h"

// GGML internal header

// #define GGML_USE_QLUTATTN
#if defined(GGML_USE_QLUTATTN)

#ifdef  __cplusplus
extern "C" {
#endif

// Declare the buffer type function that returns a pointer to the buffer type
ggml_backend_buffer_type_t ggml_backend_qlutattn_buffer_type(void);

// Initialization function
void ggml_qlutattn_init(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_USE_QLUTATTN 