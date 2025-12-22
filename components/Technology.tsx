
'use client';

import React from 'react';
import { motion } from 'framer-motion';
import styles from './Technology.module.css';
import {
    SiPython, SiMysql, SiTableau,
    SiDatabricks, SiApachespark, SiGooglecloud, SiApachekafka,
    SiKubernetes, SiDocker, SiSnowflake, SiStreamlit
} from 'react-icons/si';
import { RiFileExcel2Fill } from 'react-icons/ri';
import {
    Activity, Network, PieChart, LineChart, Sigma,
    Workflow, BarChart, FileCode
} from 'lucide-react';

const techs = [
    { name: 'Python', icon: <SiPython size={50} color="#3776AB" /> },
    { name: 'R Studio', icon: <FileCode size={50} color="#276DC3" /> },
    { name: 'SQL', icon: <SiMysql size={50} color="#4479A1" /> },
    { name: 'GeSCA', icon: <Network size={50} color="#666" />, textFallback: 'GeSCA' },
    { name: 'Stata', icon: <LineChart size={50} color="#1A5F91" />, textFallback: 'Stata' },
    { name: 'SmartPLS', icon: <Activity size={50} color="#29abe2" />, textFallback: 'PLS' },
    { name: 'SPSS', icon: <Sigma size={50} color="#CA1B21" />, textFallback: 'SPSS' },
    { name: 'SAS', icon: <PieChart size={50} color="#0072C6" />, textFallback: 'SAS' },
    { name: 'Tableau', icon: <SiTableau size={50} color="#E97627" /> },
    { name: 'Databricks', icon: <SiDatabricks size={50} color="#FF3621" /> },
    { name: 'Spark', icon: <SiApachespark size={50} color="#E25A1C" /> },
    { name: 'GCP', icon: <SiGooglecloud size={50} color="#4285F4" /> },
    { name: 'Kafka', icon: <SiApachekafka size={50} color="#231F20" /> },
    { name: 'Kubernetes', icon: <SiKubernetes size={50} color="#326CE5" /> },
    { name: 'Docker', icon: <SiDocker size={50} color="#2496ED" /> },
    { name: 'n8n', icon: <Workflow size={50} color="#FF6D5A" /> },
    { name: 'Power BI', icon: <BarChart size={50} color="#F2C811" /> },
    { name: 'Excel', icon: <RiFileExcel2Fill size={50} color="#217346" /> },
    { name: 'Snowflake', icon: <SiSnowflake size={50} color="#29B5E8" /> },
    { name: 'Streamlit', icon: <SiStreamlit size={50} color="#FF4B4B" /> },
];

import { useLanguage } from '@/context/LanguageContext';

const Technology = () => {
    const { t } = useLanguage();
    return (
        <section className={styles.section} id="technology">
            <h2 className={styles.title}>{t.technology.title}</h2>
            <div className={styles.container}>
                <div className={styles.grid}>
                    {techs.map((tech, index) => (
                        <motion.div
                            key={tech.name}
                            className={styles.card}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.05 }}
                        >
                            <div className={styles.iconWrapper}>
                                {tech.icon ? tech.icon : (
                                    <span style={{ fontSize: '1.5rem', fontWeight: 800, color: '#1a202c' }}>
                                        {tech.textFallback || tech.name}
                                    </span>
                                )}
                            </div>
                            <span className={styles.label}>{tech.name}</span>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Technology;
