'use client';

import React from 'react';
import { motion } from 'framer-motion';
import styles from './CTA.module.css';

import { useLanguage } from '@/context/LanguageContext';

const CTA = () => {
    const { t } = useLanguage();
    return (
        <section className={styles.section}>
            <motion.h2
                className={styles.heading}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
            >
                {t.cta.heading}
            </motion.h2>

            <motion.a
                href="https://beacons.ai/rsdatagroup"
                target="_blank"
                rel="noopener noreferrer"
                className={styles.ctaButton}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.2 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
            >
                {t.cta.button}
            </motion.a>
        </section>
    );
};

export default CTA;
