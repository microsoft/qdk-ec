/*
 * Minimal C consumer of the qodec ABI: open a qodec, walk its lowering chain
 * and one gadget's decoding surface, then release it.
 *
 * Note what is absent: no malloc, no free, no accessor calls. qodec_load
 * hands back the root and it is all struct traversal from there.
 *
 * Built and run by tests/c_smoke_test.rs.
 */
#include <stdio.h>

#include "qodec.h"

/* Every string list is CSR: a NUL-terminated blob plus offsets into it. */
static void print_strings(const QodecStrings *list) {
    for (size_t i = 0; i < list->count; ++i) {
        printf("%s%s", i ? " " : "", list->bytes + list->offsets[i]);
    }
}

/* QodecArgument represents both block operands and parameter arguments.
 * Its value.tag selects the supplied representation, not a declared parameter type. */
static void print_argument(const QodecArgument *a) {
    if (a->name) {
        printf(" %s=", a->name);
    } else {
        printf(" ");
    }
    switch (a->value.tag) {
    case QodecArgumentValue_Qubit:
        printf("q%llu", (unsigned long long)a->value.qubit.index);
        break;
    case QodecArgumentValue_Readout:
        printf("rec[%llu]", (unsigned long long)a->value.readout.index);
        break;
    case QodecArgumentValue_Integer:
        printf("%lld", (long long)a->value.integer.value);
        break;
    case QodecArgumentValue_Number:
        printf("%g", a->value.number.value);
        break;
    case QodecArgumentValue_Boolean:
        printf("%s", a->value.boolean.value ? "true" : "false");
        break;
    case QodecArgumentValue_Text:
        printf("%s", a->value.text.value);
        break;
    case QodecArgumentValue_QubitList:
        printf("[");
        for (size_t i = 0; i < a->value.qubit_list.qubits.count; ++i) {
            printf("%sq%llu", i ? " " : "", (unsigned long long)a->value.qubit_list.qubits.items[i]);
        }
        printf("]");
        break;
    case QodecArgumentValue_StringList:
        printf("[");
        print_strings(&a->value.string_list.strings);
        printf("]");
        break;
    default:
        printf("<unknown argument tag %u>", (unsigned)a->value.tag);
        break;
    }
}

/* The parsed circuit: a C consumer never writes a stim or YAML parser. */
static void print_program(const QodecCircuit *circuit) {
    if (circuit->error) {
        printf("  circuit did not parse: %s\n", circuit->error);
        return;
    }
    printf("  program: %zu calls\n", circuit->calls.count);
    for (size_t i = 0; i < circuit->calls.count; ++i) {
        const QodecInstructionCall *call = &circuit->calls.items[i];
        printf("    %s", call->mnemonic);
        for (size_t s = 0; s < call->operands.count; ++s) {
            print_argument(&call->operands.items[s]);
        }
        for (size_t o = 0; o < call->arguments.count; ++o) {
            print_argument(&call->arguments.items[o]);
        }
        for (size_t p = 0; p < call->select.count; ++p) {
            printf(" select{");
            for (size_t c = call->select.offsets[p]; c < call->select.offsets[p + 1]; ++c) {
                printf("%s%s=%u", c == call->select.offsets[p] ? "" : " ", call->select.constraints[c].flag,
                       (unsigned)call->select.constraints[c].bit);
            }
            printf("}");
        }
        printf("\n");
    }
}

/* QodecAction and QodecArgumentValue are tagged unions: read the arm the tag names. */
static void print_action(const QodecInstruction *instruction) {
    for (size_t s = 0; s < instruction->action.count; ++s) {
        const QodecAction *a = &instruction->action.items[s].action;
        printf("  action[%zu]: ", s);
        switch (a->tag) {
        case QodecAction_Stabilize:
            printf("stabilize ");
            print_strings(&a->stabilize.paulis);
            break;
        case QodecAction_Observe:
            printf("observe ");
            print_strings(&a->observe.observables);
            break;
        case QodecAction_Pauli:
            printf("pauli %s", a->pauli.pauli);
            break;
        case QodecAction_Clifford:
            printf("clifford");
            for (size_t i = 0; i < a->clifford.from.count; ++i) {
                printf(" %s->%s", a->clifford.from.bytes + a->clifford.from.offsets[i],
                       a->clifford.to.bytes + a->clifford.to.offsets[i]);
            }
            break;
        case QodecAction_Rotate:
            printf("rotate %s by ", a->rotate.axis);
            if (a->rotate.angle_is_literal) {
                printf("%g", a->rotate.angle_literal);
            } else {
                printf("<%s>", a->rotate.angle_operand);
            }
            break;
        default:
            printf("<unknown action tag %u>", (unsigned)a->tag);
            break;
        }
        printf("\n");
    }
}

static void print_reference(const QodecReference *r) {
    switch (r->tag) {
    case QODEC_REFERENCE_CONSTANT:
        printf(" %llu", (unsigned long long)r->index);
        break;
    case QODEC_REFERENCE_CIRCUIT_READOUT:
        printf(" circuit.readouts[%llu]", (unsigned long long)r->index);
        break;
    case QODEC_REFERENCE_READOUT:
        printf(" readouts[%llu]", (unsigned long long)r->index);
        break;
    case QODEC_REFERENCE_ENCODING_PROPERTY:
        printf(" %s[%llu].%s[%llu]", r->boundary == QODEC_BOUNDARY_IN ? "in" : "out", (unsigned long long)r->entry,
               r->property == QODEC_PROPERTY_STABILIZER  ? "stabilizers"
               : r->property == QODEC_PROPERTY_LOGICAL_X ? "x"
                                                     : "z",
               (unsigned long long)r->index);
        break;
    default:
        printf(" <unknown tag %u>", (unsigned)r->tag);
        break;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <manifest-or-bundle-file>\n", argv[0]);
        return 2;
    }

    if (qodec_abi_version() != QODEC_ABI_VERSION) {
        fprintf(stderr, "ABI mismatch: header %d, library %u\n", QODEC_ABI_VERSION, qodec_abi_version());
        return 1;
    }

    Qodec *qodec = NULL;
    if (qodec_load(argv[1], &qodec) != QODEC_STATUS_OK) {
        fprintf(stderr, "open failed: %s\n", qodec_last_error());
        return 1;
    }

    printf("qodec: %s\n", qodec->name);
    printf("layers: %zu\n", qodec->layers.count);

    for (size_t i = 0; i < qodec->layers.count; ++i) {
        const QodecLayer *layer = &qodec->layers.items[i];
        printf("layer %zu: %s (%zu blocks, %zu instructions, %zu gadgets)\n", i, layer->instruction_set_name, layer->blocks.count,
               layer->instructions.count, layer->gadgets.count);
    }

    if (qodec->layers.count == 0 || qodec->layers.items[0].gadgets.count == 0) {
        printf("no gadgets to inspect\nok\n");
        qodec_unload(qodec);
        return 0;
    }

    /* Inspect whichever gadget the top layer lists first. */
    const QodecGadget *gadget = &qodec->layers.items[0].gadgets.items[0];
    /* The formal semantics of every instruction the top ISA declares. */
    const QodecLayer *top = &qodec->layers.items[0];
    for (size_t i = 0; i < top->instructions.count; ++i) {
        const QodecInstruction *instruction = &top->instructions.items[i];
        printf("%s(%zu in, %zu out, %zu params, %zu flags)\n", instruction->mnemonic, instruction->inputs.count,
               instruction->outputs.count, instruction->parameters.count, instruction->flags.count);
        print_action(instruction);
    }

    printf("inspecting gadget: %s\n", gadget->implements.mnemonic);
    printf("  circuit targets %s (%s)\n", gadget->circuit.instruction_set_name, gadget->circuit.effective_format);
    print_program(&gadget->circuit);

    printf("checks: %zu (%zu references)\n", gadget->checks.count, gadget->checks.total);
    for (size_t i = 0; i < gadget->checks.count; ++i) {
        printf("  check %zu:", i);
        for (size_t j = gadget->checks.offsets[i]; j < gadget->checks.offsets[i + 1]; ++j) {
            print_reference(&gadget->checks.references[j]);
        }
        printf("\n");
    }

    for (size_t e = 0; e < gadget->inputs.count; ++e) {
        const QodecEncoding *encoding = &gadget->inputs.items[e];
        const QodecCode *code = &encoding->code;
        printf("in[%zu]: code %s, %zu stabilizers\n", e, code->name, code->stabilizers.count);
        for (size_t s = 0; s < code->stabilizers.count; ++s) {
            printf("  %s\n", code->stabilizers.bytes + code->stabilizers.offsets[s]);
        }
    }

    /* A deliberate miss, to show the null/last_error pairing works. */
    if (qodec_find_gadget(&qodec->layers.items[0], "no_such_instruction") != NULL) {
        fprintf(stderr, "expected NULL for an unknown mnemonic\n");
        qodec_unload(qodec);
        return 1;
    }
    printf("unknown mnemonic rejected: %s\n", qodec_last_error());

    qodec_unload(qodec);
    printf("ok\n");
    return 0;
}
