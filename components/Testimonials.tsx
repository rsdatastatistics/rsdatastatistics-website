'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Star } from 'lucide-react';
import { FcGoogle } from 'react-icons/fc';
import { Swiper, SwiperSlide } from 'swiper/react';
import { Pagination, Autoplay } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/pagination';
import { useLanguage } from '@/context/LanguageContext';
import styles from './Testimonials.module.css';

// Reviews moved to translations.ts

const Testimonials = () => {
    const { t } = useLanguage();
    return (
        <section className={styles.section}>
            <div className={styles.container}>
                <h2 className={styles.title}><span>{t.testimonials.title_prefix}</span> {t.testimonials.title_suffix}</h2>
                <Swiper
                    modules={[Pagination, Autoplay]}
                    spaceBetween={30}
                    slidesPerView={1}
                    centeredSlides={true}
                    loop={true}
                    autoplay={{
                        delay: 3000,
                        disableOnInteraction: false,
                    }}
                    pagination={{
                        clickable: true,
                        dynamicBullets: true,
                    }}
                    breakpoints={{
                        640: {
                            slidesPerView: 1,
                        },
                        768: {
                            slidesPerView: 2,
                        },
                        1024: {
                            slidesPerView: 3,
                        },
                    }}
                    className={styles.swiperContainer}
                >
                    {t.testimonials.reviews.map((review, index) => (
                        <SwiperSlide key={review.id} className={styles.slide}>
                            <motion.div
                                className={styles.card}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: 0.1 }}
                            >
                                <div className={styles.header}>
                                    <div className={styles.avatar}>{review.name[0]}</div>
                                    <div>
                                        <h4 className={styles.name}>{review.name}</h4>
                                        <div className={styles.occupation}>{review.occupation}</div>
                                        <div className={styles.stars}>
                                            {[...Array(5)].map((_, i) => ( // hardcoded rating 5 for all as per original data or we can add rating to translation if diverse
                                                <Star key={i} size={16} fill="#FFD700" color="#FFD700" />
                                            ))}
                                        </div>
                                    </div>
                                </div>
                                <p className={styles.text}>"{review.text}"</p>
                                <div className={styles.google}>
                                    <FcGoogle size={20} />
                                    <span>{t.testimonials.google_review}</span>
                                </div>
                            </motion.div>
                        </SwiperSlide>
                    ))}
                </Swiper>
            </div>
        </section>
    );
};

export default Testimonials;
