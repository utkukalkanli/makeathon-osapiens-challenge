import { afterEach, describe, expect, test } from "vitest";
import { SCROLL_LOCK_CLASS, setScrollLock } from "./scrollLock";

describe("setScrollLock", () => {
  afterEach(() => document.documentElement.classList.remove(SCROLL_LOCK_CLASS));

  test("locks and unlocks page scrolling via a class on <html>", () => {
    setScrollLock(document, true);
    expect(document.documentElement.classList.contains(SCROLL_LOCK_CLASS)).toBe(true);
    setScrollLock(document, false);
    expect(document.documentElement.classList.contains(SCROLL_LOCK_CLASS)).toBe(false);
  });

  test("unlocking an unlocked page is a no-op", () => {
    setScrollLock(document, false);
    expect(document.documentElement.classList.contains(SCROLL_LOCK_CLASS)).toBe(false);
  });
});
