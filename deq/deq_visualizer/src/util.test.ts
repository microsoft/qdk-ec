import { expect, test } from 'vitest'
import { cached_dict } from './util'

test('cached dict', () => {
  let counter = 0
  class Tester {
    bias: number
    constructor(bias: number) {
      this.bias = bias
    }
    @cached_dict
    gadgets(gid: bigint): number {
      counter += 1
      return Number(gid) + this.bias
    }
  }
  const tester1 = new Tester(1)
  const tester2 = new Tester(2)
  expect(tester1.gadgets(10n)).toBe(11)
  expect(counter).toBe(1)
  expect(tester1.gadgets(10n)).toBe(11)
  expect(counter).toBe(1)
  expect(tester1.gadgets(10n)).toBe(11)
  expect(counter).toBe(1)
  expect(tester1.gadgets(20n)).toBe(21)
  expect(counter).toBe(2)

  expect(tester2.gadgets(10n)).toBe(12)
  expect(counter).toBe(3)
  expect(tester2.gadgets(10n)).toBe(12)
  expect(counter).toBe(3)
  expect(tester2.gadgets(10n)).toBe(12)
  expect(counter).toBe(3)
  expect(tester2.gadgets(20n)).toBe(22)
  expect(counter).toBe(4)

  expect(tester1.gadgets(10n)).toBe(11)
  expect(counter).toBe(4)
})
