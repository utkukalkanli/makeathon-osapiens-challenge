/** "Orbi·tree": the split shows how to say it. */
export function Wordmark({ as: Tag = "h1" }: { as?: "h1" | "span" }) {
  return (
    <Tag className="wordmark" aria-label="Orbitree">
      <span aria-hidden="true">
        Orbi<span className="wordmark__dot">·</span>tree
      </span>
    </Tag>
  );
}
