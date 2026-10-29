// The story is driven by scroll position, so a restored scroll (reload,
// back/forward, browser session restore) would open the app mid-story or in
// explore mode. Always start at the opening chapter.
export function startAtTop(win: Pick<Window, "scrollTo"> & { history: Pick<History, "scrollRestoration"> }): void {
  win.history.scrollRestoration = "manual";
  win.scrollTo(0, 0);
}
