'use client';

import React from 'react';
import styles from './Footer.module.css';

import { useLanguage } from '@/context/LanguageContext';

const Footer = () => {
    const { t } = useLanguage();
    return (
        <footer className={styles.footer}>
            <div className={styles.container}>
                <div className={styles.column}>
                    <h3>RS Data Statistics</h3>
                    <p>{t.footer.description}</p>
                    <div style={{ marginTop: '1.5rem', fontSize: '0.95rem' }}>
                        <p style={{ marginBottom: '0.5rem' }}><strong>{t.footer.email}</strong> admin@rsdatastatistics.com</p>
                        <p><strong>{t.footer.address_label}</strong> {t.footer.address_value}</p>
                    </div>
                </div>
                <div className={styles.column}>
                    <h4>{t.footer.links}</h4>
                    <a href="/">{t.navbar.home}</a>
                    <a href="/blogs">{t.navbar.blogs}</a>
                    <a href="/careers">{t.navbar.careers}</a>
                    <a href="/about">{t.navbar.about}</a>
                </div>
                <div className={styles.column}>
                    <h4>{t.footer.location}</h4>
                    <iframe
                        src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3716527.2944453293!2d107.16254947984751!3d-7.083200849267471!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x2dcc673dd39a8bbf%3A0x54e135259d460e73!2sRS%20Data%20Statistics!5e1!3m2!1sen!2spl!4v1726608205284!5m2!1sen!2spl"
                        width="100%"
                        height="300"
                        style={{ border: 0, borderRadius: '8px' }}
                        allowFullScreen
                        loading="lazy"
                        referrerPolicy="no-referrer-when-downgrade"
                    ></iframe>
                </div>
            </div>
            <div className={styles.copyright}>
                &copy; {new Date().getFullYear()} RS Data Statistics. All rights reserved.
            </div>
        </footer>
    );
};

export default Footer;
