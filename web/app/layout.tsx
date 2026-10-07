import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "Huddle — Football Film Assistant", description: "AI assisted American football film tagging" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
