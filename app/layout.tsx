import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Orion — Inteligencia deportiva",
  description:
    "Agente personal de inteligencia deportiva con ejecución local y control explícito de recursos y privacidad.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
