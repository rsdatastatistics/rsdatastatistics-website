import React from 'react';
import { notFound } from 'next/navigation';
import { getPostData } from '@/lib/posts';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import styles from './BlogPost.module.css';

import { Metadata } from 'next';

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
    const { slug } = await params;
    const post = getPostData(slug);

    if (!post) {
        return {
            title: 'Post Not Found',
        };
    }

    return {
        title: `${post.title} | RS Data Statistics`,
        description: post.content.substring(0, 160) + '...',
        openGraph: {
            title: post.title,
            description: post.content.substring(0, 160) + '...',
            type: 'article',
            publishedTime: post.date,
            images: post.image ? [post.image] : [],
        },
    };
}

export default async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
    const { slug } = await params;
    const post = getPostData(slug);

    if (!post) {
        notFound();
    }

    return (
        <article className={styles.article}>
            <header className={styles.header}>
                <h1 className={styles.title}>{post.title}</h1>
                <div className={styles.meta}>
                    <time>{post.date}</time>
                </div>
            </header>
            <div className={styles.content}>
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw]}
                    components={{
                        code({ node, inline, className, children, ...props }: any) {
                            const match = /language-(\w+)/.exec(className || '')
                            return !inline && match ? (
                                <SyntaxHighlighter
                                    style={vscDarkPlus}
                                    language={match[1]}
                                    PreTag="div"
                                    {...props}
                                >
                                    {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                            ) : (
                                <code className={className} {...props}>
                                    {children}
                                </code>
                            )
                        },
                        // Fix for React vAlign error by filtering out invalid props
                        th: ({ node, vAlign, ...props }: any) => <th {...props} />,
                        td: ({ node, vAlign, ...props }: any) => <td {...props} />,
                        tr: ({ node, vAlign, ...props }: any) => <tr {...props} />,
                    }}
                >
                    {post.content}
                </ReactMarkdown>
            </div>
        </article>
    );
}
