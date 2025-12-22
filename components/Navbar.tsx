'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { Menu, X } from 'lucide-react';
import styles from './Navbar.module.css';

import { useLanguage } from '@/context/LanguageContext';

const Navbar = () => {
    const { t, language, setLanguage } = useLanguage();
    const [isOpen, setIsOpen] = useState(false);

    const toggleMenu = () => setIsOpen(!isOpen);

    return (
        <nav className={styles.navbar}>
            <div className={styles.container}>
                <Link href="/" className={styles.logo}>
                    <Image
                        src="/logo.png"
                        alt="RS Data Statistics Logo"
                        width={40}
                        height={40}
                        style={{ objectFit: 'contain' }}
                    />
                    RS Data Statistics
                </Link>
                <div className={styles.desktopMenu}>
                    <Link href="/" className={styles.link}>{t.navbar.home}</Link>
                    <Link href="/blogs" className={styles.link}>{t.navbar.blogs}</Link>
                    <Link href="/careers" className={styles.link}>{t.navbar.careers}</Link>
                    <Link href="/about" className={styles.link}>{t.navbar.about}</Link>
                    <button
                        onClick={() => setLanguage(language === 'en' ? 'id' : 'en')}
                        className={styles.langBtn}
                        style={{
                            marginLeft: '20px',
                            padding: '5px 10px',
                            cursor: 'pointer',
                            background: 'transparent',
                            border: '1px solid currentColor',
                            borderRadius: '4px',
                            color: 'inherit',
                            fontSize: '0.9rem',
                            fontWeight: 'bold'
                        }}
                    >
                        {language === 'en' ? 'ID' : 'EN'}
                    </button>
                </div>
                <button className={styles.mobileToggle} onClick={toggleMenu}>
                    {isOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>
            {isOpen && (
                <div className={styles.mobileMenu}>
                    <Link href="/" className={styles.mobileLink} onClick={toggleMenu}>{t.navbar.home}</Link>
                    <Link href="/blogs" className={styles.mobileLink} onClick={toggleMenu}>{t.navbar.blogs}</Link>
                    <Link href="/careers" className={styles.mobileLink} onClick={toggleMenu}>{t.navbar.careers}</Link>
                    <Link href="/about" className={styles.mobileLink} onClick={toggleMenu}>{t.navbar.about}</Link>
                    <button
                        onClick={() => {
                            setLanguage(language === 'en' ? 'id' : 'en');
                            toggleMenu();
                        }}
                        className={styles.mobileLink}
                        style={{
                            textAlign: 'left',
                            background: 'none',
                            border: 'none',
                            cursor: 'pointer',
                            color: 'inherit',
                            fontSize: '1.1rem',
                            fontWeight: 500,
                            padding: '1rem 2rem'
                        }}
                    >
                        Switch to {language === 'en' ? 'Indonesia' : 'English'}
                    </button>
                </div>
            )}
        </nav>
    );
};

export default Navbar;
