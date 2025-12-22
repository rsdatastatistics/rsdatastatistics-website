'use client';

import React from 'react';
import Link from 'next/link';
import { useLanguage } from '@/context/LanguageContext';
import styles from './About.module.css';

export default function AboutPage() {
    const { t } = useLanguage();
    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <h1 className={styles.title}>{t.about.title}</h1>
                <p className={styles.subtitle}>{t.about.subtitle}</p>
            </header>

            <div className={styles.content}>
                <section className={styles.section}>
                    <h2>{t.about.who_we_are_title}</h2>
                    <p>
                        {t.about.who_we_are_text}
                    </p>
                </section>

                <section className={styles.section}>
                    <h2>{t.about.mission_title}</h2>
                    <p>
                        {t.about.mission_text}
                    </p>
                </section>

                <section className={styles.section}>
                    <h2>{t.about.why_choose_title}</h2>
                    <ul className={styles.list}>
                        {t.about.why_choose_list.map((item, index) => (
                            <li key={index} dangerouslySetInnerHTML={{ __html: item }} />
                        ))}
                    </ul>
                    <div className={styles.buttonContainer}>
                        <Link href="https://beacons.ai/rsdatagroup" target="_blank" rel="noopener noreferrer" className={styles.ctaButton}>
                            {t.cta.button}
                        </Link>
                    </div>
                </section>
            </div>
        </div>
    );
}
