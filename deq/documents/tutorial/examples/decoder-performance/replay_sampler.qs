namespace ReplaySampler {
    operation ApplyBody(
        qubits : Qubit[],
        mapping : Int[],
        instructions : Int[][],
        measurementCount : Int
    ) : Result[] {
        mutable results = [Zero, size=measurementCount];
        mutable measured = 0;
        for instruction in instructions {
            let opcode = instruction[0];
            if opcode == 6 or opcode == 7 or opcode == 13 {
                for index in 1..2..Length(instruction)-1 {
                    let first = qubits[mapping[instruction[index]]];
                    let second = qubits[mapping[instruction[index+1]]];
                    if opcode == 6 {
                        CNOT(first, second);
                    } elif opcode == 7 {
                        CZ(first, second);
                    } else {
                        Rxx(0.0, first, second);
                    }
                }
            } else {
                for index in 1..Length(instruction)-1 {
                    let qubit = qubits[mapping[instruction[index]]];
                    if opcode == 0 {
                        Reset(qubit);
                    } elif opcode == 1 {
                        Reset(qubit);
                        H(qubit);
                    } elif opcode == 2 {
                        X(qubit);
                    } elif opcode == 3 {
                        Y(qubit);
                    } elif opcode == 4 {
                        Z(qubit);
                    } elif opcode == 5 {
                        H(qubit);
                    } elif opcode == 8 or opcode == 14 {
                        if opcode == 8 {
                            Rz(0.0, qubit);
                        }
                        set results w/= measured <- M(qubit);
                        set measured += 1;
                    } elif opcode == 9 or opcode == 15 {
                        if opcode == 9 {
                            Rx(0.0, qubit);
                        }
                        set results w/= measured <- Measure([PauliX], [qubit]);
                        set measured += 1;
                    } elif opcode == 10 {
                        Rz(0.0, qubit);
                    } elif opcode == 11 {
                        Rx(0.0, qubit);
                    } elif opcode == 12 {
                        Ry(0.0, qubit);
                    } else {
                        fail "Unsupported physical instruction";
                    }
                }
            }
        }
        if measured != measurementCount {
            fail "Gadget measurement count differs from decoder metadata";
        }
        return results;
    }

    operation Sample(
        program : Int[][],
        inputs : Int[][][],
        outputs : Int[][][],
        usedQubits : Int[][],
        retiredQubits : Int[][],
        bodies : Int[][][],
        measurementCounts : Int[],
        virtualGadgets : Bool[],
        qubitCount : Int,
        measurementCount : Int
    ) : Int[] {
        use qubits = Qubit[qubitCount];
        mutable firstOutputs = [[0, size=0], size=Length(program)+1];
        mutable secondOutputs = [[0, size=0], size=Length(program)+1];
        mutable freeQubits = [0, size=qubitCount];
        mutable freeCount = 0;
        mutable allocated = 0;
        mutable words = [0, size=(measurementCount+63)/64];
        mutable measured = 0;
        for position in 0..Length(program)-1 {
            let descriptor = program[position];
            let kind = descriptor[0];
            let gid = position+1;
            if virtualGadgets[kind] {
                if descriptor[1] == 0 or descriptor[3] != 0 {
                    fail "Invalid virtual passthrough";
                }
                if descriptor[2] == 0 {
                    set firstOutputs w/= gid <- firstOutputs[descriptor[1]];
                    set firstOutputs w/= descriptor[1] <- [];
                } else {
                    set firstOutputs w/= gid <- secondOutputs[descriptor[1]];
                    set secondOutputs w/= descriptor[1] <- [];
                }
            } else {
                let used = usedQubits[kind];
                mutable mapping = [-1, size=used[Length(used)-1]+1];
                for port in 0..Length(inputs[kind])-1 {
                    let parent = descriptor[1+2*port];
                    let parentPort = descriptor[2+2*port];
                    let physical = parentPort == 0
                        ? firstOutputs[parent] | secondOutputs[parent];
                    let local = inputs[kind][port];
                    if Length(physical) != Length(local) {
                        fail "Physical input port size mismatch";
                    }
                    for index in 0..Length(local)-1 {
                        set mapping w/= local[index] <- physical[index];
                    }
                    if parentPort == 0 {
                        set firstOutputs w/= parent <- [];
                    } else {
                        set secondOutputs w/= parent <- [];
                    }
                }
                for local in used {
                    if mapping[local] < 0 {
                        if freeCount > 0 {
                            set freeCount -= 1;
                            set mapping w/= local <- freeQubits[freeCount];
                        } else {
                            if allocated >= qubitCount {
                                fail "Physical qubit allocation exceeded the qualified bound";
                            }
                            set mapping w/= local <- allocated;
                            set allocated += 1;
                        }
                    }
                }
                let outcomes = ApplyBody(qubits, mapping, bodies[kind], measurementCounts[kind]);
                for outcome in outcomes {
                    if outcome == One {
                        set words w/= measured/64 <- words[measured/64] ||| (1 <<< (measured%64));
                    }
                    set measured += 1;
                }
                for port in 0..Length(outputs[kind])-1 {
                    let local = outputs[kind][port];
                    mutable physical = [0, size=Length(local)];
                    for index in 0..Length(local)-1 {
                        set physical w/= index <- mapping[local[index]];
                    }
                    if port == 0 {
                        set firstOutputs w/= gid <- physical;
                    } else {
                        set secondOutputs w/= gid <- physical;
                    }
                }
                for local in retiredQubits[kind] {
                    let physical = mapping[local];
                    Reset(qubits[physical]);
                    set freeQubits w/= freeCount <- physical;
                    set freeCount += 1;
                }
            }
        }
        if measured != measurementCount {
            fail "Program measurement count differs from decoder metadata";
        }
        ResetAll(qubits);
        return words;
    }
}
