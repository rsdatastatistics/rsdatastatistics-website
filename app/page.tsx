'use client';

import React from 'react';
import Hero3D from '@/components/Hero3D';
import ServiceCarousel from '@/components/ServiceCarousel';
import Technology from '@/components/Technology';
import Testimonials from '@/components/Testimonials';
import CTA from '@/components/CTA';

export default function Home() {
    return (
        <main style={{ width: '100%', overflowX: 'hidden' }}>
            <Hero3D />
            <ServiceCarousel />
            <Technology />
            <Testimonials />
            <CTA />
        </main>
    );
}
