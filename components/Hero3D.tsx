'use client';

import React from 'react';
import { motion } from 'framer-motion';
import styles from './Hero3D.module.css';

import { useLanguage } from '@/context/LanguageContext';

const Hero3D = () => {
    const { t } = useLanguage();
    return (
        <section className={styles.hero}>
            <div className={styles.content}>
                <motion.span
                    className={styles.eyebrow}
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    {t.hero.eyebrow}
                </motion.span>

                <motion.h1
                    className={styles.title}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                >
                    {t.hero.title} <br />
                    {t.hero.subtitle}
                    <span className={styles.numberOne}>#1</span>
                </motion.h1>

                <motion.p
                    className={styles.description}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.4 }}
                >
                    {t.hero.description}
                </motion.p>

                <motion.a
                    href="https://beacons.ai/rsdatagroup"
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.ctaButton}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.6 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    {t.hero.cta}
                </motion.a>
            </div>
        </section>
    );
};

export default Hero3D;
