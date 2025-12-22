import { MetadataRoute } from 'next';
import { getSortedPostsData } from '@/lib/posts';

export default function sitemap(): MetadataRoute.Sitemap {
    const posts = getSortedPostsData();
    const baseUrl = 'https://rsdatastatistics.com';

    const blogEntries = posts.map((post) => ({
        url: `${baseUrl}/blogs/${post.slug}`,
        lastModified: new Date(post.date),
        changeFrequency: 'monthly' as const,
        priority: 0.8,
    }));

    return [
        {
            url: baseUrl,
            lastModified: new Date(),
            changeFrequency: 'yearly',
            priority: 1,
        },
        {
            url: `${baseUrl}/blogs`,
            lastModified: new Date(),
            changeFrequency: 'weekly',
            priority: 0.9,
        },
        {
            url: `${baseUrl}/about`,
            lastModified: new Date(),
            changeFrequency: 'yearly',
            priority: 0.8,
        },
        {
            url: `${baseUrl}/careers`,
            lastModified: new Date(),
            changeFrequency: 'monthly',
            priority: 0.8,
        },
        ...blogEntries,
    ];
}
