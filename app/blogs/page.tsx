
import { getSortedPostsData } from '@/lib/posts';
import BlogPageContent from '@/components/BlogPageContent';

// Server Component
export default async function BlogPage() {
    const posts = getSortedPostsData();

    return (
        <BlogPageContent posts={posts} />
    );
}

