/** Public repo (from `git remote get-url origin`, as https). */
export const REPO_URL = "https://github.com/utkukalkanli/makeathon-osapiens-challenge";

export function Credits({ className = "" }: { className?: string }) {
  return (
    <p className={`credits ${className}`.trim()}>
      <span>osapiens Terra Makeathon 2026</span>
      <span className="credits__sep" aria-hidden="true">·</span>
      <span>Built by Utku Kalkanlı</span>
      <span className="credits__sep" aria-hidden="true">·</span>
      <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
        GitHub
      </a>
    </p>
  );
}

export default Credits;
