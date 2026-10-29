import { describe, expect, test, vi } from "vitest";
import { startAtTop } from "./startAtTop";

describe("startAtTop", () => {
  test("disables browser scroll restoration and jumps to the top", () => {
    const win = {
      history: { scrollRestoration: "auto" as ScrollRestoration },
      scrollTo: vi.fn(),
    };
    startAtTop(win);
    expect(win.history.scrollRestoration).toBe("manual");
    expect(win.scrollTo).toHaveBeenCalledWith(0, 0);
  });
});
