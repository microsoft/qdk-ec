/* Uses only the installed qodec package. Compiled as both C and C++. */
#include <stdio.h>
#include <string.h>

#include <qodec.h>

#include "consumer_module.h"

static int fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    return 1;
}

static int check_repetition3(const Qodec *qodec) {
    if (qodec->layers.count != 2) {
        return fail("expected two layers");
    }
    const QodecLayer *layer = &qodec->layers.items[0];
    if (strcmp(layer->instruction_set_name, "repetition3") != 0 || layer->gadgets.count != 4) {
        return fail("unexpected logical layer");
    }
    if (qodec_find_gadget(layer, "measure_z") == NULL || qodec_find_gadget(layer, "missing") != NULL) {
        return fail("unexpected gadget lookup");
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        return fail("usage: consumer <repetition3.qodec.yaml> <missing path>");
    }
    if (qodec_abi_version() != QODEC_ABI_VERSION || consumer_module_abi_version() != QODEC_ABI_VERSION) {
        return fail("qodec ABI mismatch");
    }

    Qodec *qodec = NULL;
    if (qodec_load(argv[1], &qodec) != QODEC_STATUS_OK) {
        const char *message = qodec_last_error();
        return fail(message != NULL ? message : "qodec load failed");
    }
    int result = check_repetition3(qodec);
    qodec_unload(qodec);
    if (result != 0) {
        return result;
    }

    Qodec *missing = NULL;
    if (qodec_load(argv[2], &missing) != QODEC_STATUS_ERROR || missing != NULL || qodec_last_error() == NULL) {
        return fail("loading a missing path did not report an error");
    }
    printf("qodec consumer: ok\n");
    return 0;
}
