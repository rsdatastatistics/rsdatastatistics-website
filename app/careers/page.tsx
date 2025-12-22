'use client';

import React from 'react';
import { useLanguage } from '@/context/LanguageContext';
import styles from './Careers.module.css';

export default function CareersPage() {
    const { t } = useLanguage();
    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <h1 className={styles.title}>{t.careers.title}</h1>
                <p className={styles.subtitle}>{t.careers.subtitle}</p>
            </header>

            <div className={styles.emptyState}>
                <h2>{t.careers.no_positions_title}</h2>
                <p>
                    {t.careers.no_positions_text_part1}
                    <strong> admin@rsdatastatistics.com</strong> {t.careers.no_positions_text_part2}
                </p>
            </div>
        </div>
    );
}
