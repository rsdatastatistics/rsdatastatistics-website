
'use client';

import React from 'react';
import { motion } from 'framer-motion';
import styles from './ServiceCarousel.module.css';
import { BarChart2, Search, Brain, Presentation, Code, Database } from 'lucide-react';

// services moved to translations.ts

import { useLanguage } from '@/context/LanguageContext';

const ServiceCarousel = () => {
    const { t } = useLanguage();

    // Mapping icons to service IDs from translations
    const getIcon = (id: number) => {
        switch (id) {
            case 1: return <Presentation size={32} />;
            case 2: return <Search size={32} />;
            case 3: return <BarChart2 size={32} />;
            case 4: return <Brain size={32} />;
            case 5: return <Code size={32} />;
            case 6: return <Database size={32} />;
            default: return <Presentation size={32} />;
        }
    };
    return (
        <section className={styles.section} id="services">
            <div className={styles.container}>
                <div className={styles.header}>
                    <h2 className={styles.title}><span>{t.services.title_prefix}</span> {t.services.title_suffix}</h2>
                    <p className={styles.subtitle}>
                        {t.services.subtitle}
                    </p>
                </div>

                <div className={styles.grid}>
                    {t.services.list.map((service, index) => (
                        <motion.div
                            key={service.id}
                            className={styles.card}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                        >
                            <div className={styles.iconWrapper}>{getIcon(service.id)}</div>
                            <h3>{service.title}</h3>
                            <p>{service.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default ServiceCarousel;
