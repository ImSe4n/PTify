/**
 * Word-by-word heading reveal.
 *
 * Borrowed from the sphericalwaves reference, which splits its headings per
 * WORD and staggers them in. Per-word rather than per-character on purpose:
 * character-level reveals are the tell of a site that is showing off, and the
 * reference itself splits by word.
 *
 * WHY THIS TAKES `string | ReactNode[]` AND NOT ARBITRARY JSX
 *
 * The obvious implementation splits `children` on whitespace. That destroys
 * markup: `AuthScreen`'s headline is
 *
 *     Notes you can read, edit, and&nbsp;<em>trust</em>.
 *
 * where a naive split loses both the non-breaking space and the emphasis. A
 * general JSX word-splitter that handles it is a 200-line utility with its own
 * bugs. Passing an array of nodes for the one heading that needs it costs a
 * single line at the call site, so that is the seam.
 *
 * ACCESSIBILITY. Splitting a sentence into spans breaks word boundaries for
 * some screen readers, which then read it letter-blocked or run words together.
 * The wrapper therefore carries the plain text as an aria-label and the spans
 * are hidden from the accessibility tree.
 */

import {
  Children,
  Fragment,
  isValidElement,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

export interface RevealProps {
  /** A plain string, or pre-split pieces when the text carries markup. */
  children: string | ReactNode[];
  /** What a screen reader should hear. Required when passing nodes. */
  label?: string;
  /** Element to render as. Headings should pass their own. */
  as?: "h1" | "h2" | "p" | "span";
  className?: string;
  /**
   * Wait until the element is scrolled into view. Off by default: above-the-
   * fold headings should animate on mount, and an observer on something already
   * visible just adds a frame of delay.
   */
  onView?: boolean;
}

/** Plain text of a node tree, for the aria-label. */
function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (isValidElement(node)) {
    return textOf((node.props as { children?: ReactNode }).children);
  }
  return "";
}

export function Reveal({
  children,
  label,
  as: Tag = "span",
  className,
  onView = false,
}: RevealProps) {
  const ref = useRef<HTMLElement>(null);
  const [shown, setShown] = useState(!onView);

  useEffect(() => {
    if (!onView || shown) return;
    const el = ref.current;
    if (!el) return;

    // The reference triggers at `top 80%`; the same point expressed as a
    // bottom-side root margin.
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { rootMargin: "0px 0px -20% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [onView, shown]);

  const pieces: ReactNode[] =
    typeof children === "string" ? children.split(/\s+/).filter(Boolean) : Children.toArray(children);

  const text = label ?? textOf(children);

  return (
    <Tag
      ref={ref as never}
      className={`reveal${shown ? " is-shown" : ""}${className ? " " + className : ""}`}
      aria-label={text}
    >
      {pieces.map((piece, i) => (
        // Two spans per word: the outer masks, the inner moves. A single span
        // would let the word slide over its neighbours instead of rising out
        // from behind the line.
        //
        // The literal space BETWEEN words is deliberate and load-bearing: a CSS
        // margin would indent every wrapped line, because it survives the line
        // break. See the note beside `.reveal` in app.css.
        <Fragment key={i}>
          {i > 0 && " "}
          <span className="rv-word" aria-hidden="true" style={{ "--i": i } as React.CSSProperties}>
            <span className="rv-inner">{piece}</span>
          </span>
        </Fragment>
      ))}
    </Tag>
  );
}
