'use client';

import React from 'react';
import Link from 'next/link';
import { useLanguage } from '@/context/LanguageContext';
import styles from '@/app/blogs/Blog.module.css';

interface PostData {
    slug: string;
    title: string;
    date: string;
    image?: string;
    content: string;
}

interface BlogPageContentProps {
    posts: PostData[];
}

const BlogPageContent = ({ posts }: BlogPageContentProps) => {
    const { t } = useLanguage();

    return (
        <div className={styles.container}>
            <header className={styles.header}>
                <h1 className={styles.title}>{t.blogs.title}</h1>
                <p className={styles.subtitle}>{t.blogs.subtitle}</p>
            </header>

            <div className={styles.grid}>
                {posts.map((post) => (
                    <Link href={`/blogs/${post.slug}`} key={post.slug} className={styles.card}>
                        {post.image && (
                            <div className={styles.cardImageContainer} style={{ width: '100%', height: '200px', position: 'relative' }}>
                                <img
                                    src={post.image}
                                    alt={post.title}
                                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                />
                            </div>
                        )}
                        <div className={styles.cardContent}>
                            <h2 className={styles.cardTitle}>{post.title}</h2>
                            <p className={styles.excerpt}>{post.content.substring(0, 150)}...</p>
                            <div className={styles.meta}>
                                <span>{post.date}</span>
                            </div>
                        </div>
                    </Link>
                ))}
                {posts.length === 0 && (
                    <div className={styles.empty}>
                        <p>{t.blogs.empty}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default BlogPageContent;
