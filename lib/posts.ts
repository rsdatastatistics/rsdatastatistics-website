import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import toml from 'toml';

const postsDirectory = path.join(process.cwd(), 'content/post');

export interface PostData {
    slug: string;
    title: string;
    date: string;
    image?: string;
    content: string;
    [key: string]: any;
}

export function getSortedPostsData(): PostData[] {
    // Create directory if it doesn't exist
    if (!fs.existsSync(postsDirectory)) {
        return [];
    }

    const fileNames = fs.readdirSync(postsDirectory);
    const allPostsData = fileNames.map((fileName) => {
        // Remove ".md" from file name to get id
        const slug = fileName.replace(/\.md$/, '');

        // Read markdown file as string
        const fullPath = path.join(postsDirectory, fileName);
        const fileContents = fs.readFileSync(fullPath, 'utf8');

        // Use gray-matter to parse the post metadata section
        const matterResult = matter(fileContents, {
            engines: {
                toml: toml.parse.bind(toml),
            },
            language: 'toml',
            delimiters: '+++'
        });

        return {
            slug,
            ...(matterResult.data as { title: string; date: string; image?: string }),
            content: matterResult.content,
        };
    });

    // Sort posts by date
    return allPostsData.sort((a, b) => {
        if (a.date < b.date) {
            return 1;
        } else {
            return -1;
        }
    });
}

export function getPostData(slug: string): PostData | null {
    const fullPath = path.join(postsDirectory, `${slug}.md`);

    if (!fs.existsSync(fullPath)) {
        return null;
    }

    const fileContents = fs.readFileSync(fullPath, 'utf8');

    // Use gray-matter to parse the post metadata section
    const matterResult = matter(fileContents, {
        engines: {
            toml: toml.parse.bind(toml),
        },
        language: 'toml',
        delimiters: '+++'
    });

    return {
        slug,
        ...(matterResult.data as { title: string; date: string; image?: string }),
        content: matterResult.content,
    };
}
