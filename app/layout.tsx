import './globals.css';
import type { Metadata } from 'next';
import { Inter, Outfit } from 'next/font/google';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import Providers from './providers';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const outfit = Outfit({ subsets: ['latin'], variable: '--font-outfit' });

export const metadata: Metadata = {
    title: 'RS Data Statistics - Jasa Analisis Data Terpercaya',
    description: 'Jasa Olah Analisis Data Terpercaya di Indonesia #1',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className={`${inter.variable} ${outfit.variable}`} suppressHydrationWarning>
                <Providers>
                    <Navbar />
                    {children}
                    <Footer />
                </Providers>
            </body>
        </html>
    );
}
