// index.html ships <html class="scroll-locked"> so the page can't scroll from
// the first paint; the app unlocks it once the loading overlay is done.
export const SCROLL_LOCK_CLASS = "scroll-locked";

export function setScrollLock(doc: Document, locked: boolean): void {
  doc.documentElement.classList.toggle(SCROLL_LOCK_CLASS, locked);
}
