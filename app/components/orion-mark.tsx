type OrionMarkProps = {
  className?: string;
  title?: string;
};

/**
 * The Orion constellation, drawn from its real star positions.
 *
 * Seven named stars: Betelgeuse and Bellatrix (shoulders), Alnitak, Alnilam
 * and Mintaka (the belt), Saiph and Rigel (feet). Rendered as SVG rather than
 * an image so it stays sharp at every size and inherits the surrounding text
 * colour, which a raster logo cannot do.
 */
export function OrionMark({ className, title }: OrionMarkProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 32 32"
      fill="none"
      role={title ? "img" : "presentation"}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      <g
        stroke="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        opacity="0.4"
      >
        {/* shoulders */}
        <path d="M9.5 8.5 L21 6.5" />
        {/* left side down to the belt */}
        <path d="M9.5 8.5 L13.5 17" />
        {/* right side down to the belt */}
        <path d="M21 6.5 L18.5 14.6" />
        {/* the belt itself */}
        <path d="M13.5 17 L16 15.8 L18.5 14.6" />
        {/* belt down to the feet */}
        <path d="M13.5 17 L11 25.5" />
        <path d="M18.5 14.6 L22 25" />
      </g>

      {/* Betelgeuse and Rigel are the brightest, so they read largest */}
      <circle cx="9.5" cy="8.5" r="2.1" fill="currentColor" />
      <circle cx="22" cy="25" r="2.1" fill="currentColor" />
      <circle cx="21" cy="6.5" r="1.5" fill="currentColor" />
      <circle cx="11" cy="25.5" r="1.5" fill="currentColor" />

      {/* the belt, evenly bright */}
      <circle cx="13.5" cy="17" r="1.25" fill="currentColor" />
      <circle cx="16" cy="15.8" r="1.25" fill="currentColor" />
      <circle cx="18.5" cy="14.6" r="1.25" fill="currentColor" />
    </svg>
  );
}
