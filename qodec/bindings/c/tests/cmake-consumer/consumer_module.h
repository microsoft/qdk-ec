#ifndef CONSUMER_MODULE_H
#define CONSUMER_MODULE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Calls qodec from a separate library; see CMakeLists.txt for why. */
uint32_t consumer_module_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif
