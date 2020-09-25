/**
 * Test module for Random tensor creation
 * TODO: Check why a.shape is of type object, instead of type Array
 * TODO: Implement toBeInstanceOf checks for data children of objects
 */
import {describe, expect, test} from '@jest/globals'

const torch = require("../dist");

describe('Random tensor creation', () => {
	test('Random tensor creation using variable number of arguements', () => {
		const a = torch.rand(1, 5).toObject();
		expect(a.data.length).toBe(5);
		expect(a.shape).toMatchObject([1,5]);

		const b = torch.rand(2, 5).toObject();
		expect(b.data.length).toBe(10);
		expect(b.shape).toMatchObject([2,5]);

		const c = torch.rand(2, 3).toObject();
		expect(c.data.length).toBe(6);
		expect(c.shape).toMatchObject([2,3]);
	})

	test('Random tensor creation using shape array', () => {
		const a = torch.rand([1, 5]).toObject();
		expect(a.data.length).toBe(5);
		expect(a.shape).toMatchObject([1,5]);

		const b = torch.rand([2, 5]).toObject();
		expect(b.data.length).toBe(10);
		expect(b.shape).toMatchObject([2,5]);

		const c = torch.rand([2, 3]).toObject();
		expect(c.data.length).toBe(6);
		expect(c.shape).toMatchObject([2,3]);
	})

	test('Random tensor creation using option parsing', () => {
		const a = torch.rand([1, 5], {
		    dtype: torch.float64
		}).toObject();
		expect(a.data.length).toBe(5);
		expect(a.shape).toMatchObject([1,5]);
	})

	test('Random tensor creation using invalid params', () => {
		// TODO
		expect(true).toBe(true);
	})
})