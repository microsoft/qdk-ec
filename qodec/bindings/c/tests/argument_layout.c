#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "qodec.h"

// Layout pins for ABI 1.
//
// A generated header cannot contradict itself, so comparing a struct with a
// copy of its own typedefs proves nothing: appending a field would update both
// sides and stay green. These are literal sizes and offsets measured once for
// ABI 1, so any layout change fails here until QODEC_ABI_VERSION is bumped.
// The literals describe a 64-bit target, which is what every shipped wheel and
// the staticlib target; a narrower target skips them and keeps the rest.

_Static_assert(QODEC_ABI_VERSION == 1, "layout test describes ABI revision 1");

_Static_assert(QodecArgumentValue_Qubit == 0, "Qubit tag changed");
_Static_assert(QodecArgumentValue_QubitList == 1, "QubitList tag changed");
_Static_assert(QodecArgumentValue_Integer == 2, "Integer tag changed");
_Static_assert(QodecArgumentValue_Number == 3, "Number tag changed");
_Static_assert(QodecArgumentValue_Text == 4, "Text tag changed");
_Static_assert(QodecArgumentValue_StringList == 5, "StringList tag changed");
_Static_assert(QodecArgumentValue_Readout == 6, "Readout tag changed");
_Static_assert(QodecArgumentValue_Boolean == 7, "Boolean tag changed");

_Static_assert(QodecAction_Stabilize == 0, "Stabilize tag changed");
_Static_assert(QodecAction_Clifford == 1, "Clifford tag changed");
_Static_assert(QodecAction_Pauli == 2, "Pauli tag changed");
_Static_assert(QodecAction_Observe == 3, "Observe tag changed");
_Static_assert(QodecAction_Rotate == 4, "Rotate tag changed");

_Static_assert(sizeof(QodecArgumentValue_Boolean_Body) == sizeof(bool), "Boolean must carry a C bool");

#if UINTPTR_MAX == 0xFFFFFFFFFFFFFFFFu

_Static_assert(sizeof(Qodec) == 48, "Qodec size changed");
_Static_assert(sizeof(QodecLayer) == 72, "QodecLayer size changed");
_Static_assert(sizeof(QodecGadget) == 448, "QodecGadget size changed");
_Static_assert(sizeof(QodecCode) == 136, "QodecCode size changed");
_Static_assert(sizeof(QodecCircuit) == 56, "QodecCircuit size changed");
_Static_assert(sizeof(QodecInstruction) == 120, "QodecInstruction size changed");
_Static_assert(sizeof(QodecActionStep) == 120, "QodecActionStep size changed");
_Static_assert(sizeof(QodecAction) == 72, "QodecAction size changed");
_Static_assert(sizeof(QodecStrings) == 32, "QodecStrings size changed");
_Static_assert(sizeof(QodecIndices) == 16, "QodecIndices size changed");
_Static_assert(sizeof(QodecParity) == 32, "QodecParity size changed");
_Static_assert(sizeof(QodecReference) == 24, "QodecReference size changed");
_Static_assert(sizeof(QodecArgument) == 48, "QodecArgument size changed");
_Static_assert(sizeof(QodecArgumentValue) == 40, "QodecArgumentValue size changed");

_Static_assert(offsetof(QodecStrings, count) == 0, "QodecStrings.count moved");
_Static_assert(offsetof(QodecStrings, offsets) == 8, "QodecStrings.offsets moved");
_Static_assert(offsetof(QodecStrings, bytes) == 16, "QodecStrings.bytes moved");
_Static_assert(offsetof(QodecStrings, total) == 24, "QodecStrings.total moved");
_Static_assert(offsetof(QodecIndices, count) == 0, "QodecIndices.count moved");
_Static_assert(offsetof(QodecIndices, items) == 8, "QodecIndices.items moved");
_Static_assert(offsetof(QodecParity, count) == 0, "QodecParity.count moved");
_Static_assert(offsetof(QodecReference, tag) == 0, "QodecReference.tag moved");
_Static_assert(offsetof(QodecReference, entry) == 8, "QodecReference.entry moved");
_Static_assert(offsetof(QodecReference, index) == 16, "QodecReference.index moved");
_Static_assert(offsetof(QodecArgument, name) == 0, "QodecArgument.name moved");
_Static_assert(offsetof(QodecArgument, value) == 8, "QodecArgument.value moved");
_Static_assert(offsetof(QodecArgumentValue, tag) == 0, "QodecArgumentValue.tag moved");
_Static_assert(offsetof(QodecAction, tag) == 0, "QodecAction.tag moved");

#endif

int main(void) {
    printf("value size=%zu align=%zu payload=%zu; argument size=%zu align=%zu value=%zu; Boolean tag=%u\n",
           sizeof(QodecArgumentValue), _Alignof(QodecArgumentValue), offsetof(QodecArgumentValue, boolean.value),
           sizeof(QodecArgument), _Alignof(QodecArgument), offsetof(QodecArgument, value),
           (unsigned)QodecArgumentValue_Boolean);
    return 0;
}