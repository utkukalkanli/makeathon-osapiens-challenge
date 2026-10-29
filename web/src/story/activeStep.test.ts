import { expect, test } from "vitest";
import { pickActiveStep } from "./activeStep";

test("pickActiveStep returns the step nearest the viewport anchor", () => {
  const r = [
    { id: "a", top: -500, bottom: -100 },
    { id: "b", top: 200, bottom: 700 },
    { id: "c", top: 900, bottom: 1400 },
  ];
  expect(pickActiveStep(r, 400)).toBe("b");
  expect(pickActiveStep(r, 800)).toBe("b");
  expect(pickActiveStep(r, 860)).toBe("c");
  expect(pickActiveStep(r, -1000)).toBe("a");
  expect(pickActiveStep(r, 5000)).toBe("c");
  expect(pickActiveStep([], 400)).toBeNull();
});
