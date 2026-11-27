#include <qodec.h>

#include "consumer_module.h"

uint32_t consumer_module_abi_version(void) {
    return qodec_abi_version();
}
