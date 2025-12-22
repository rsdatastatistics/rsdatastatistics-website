'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { translations } from '@/lib/translations';

type Language = 'en' | 'id';

interface LanguageContextType {
    language: Language;
    t: typeof translations.en;
    setLanguage: (lang: Language) => void;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const LanguageProvider = ({ children }: { children: ReactNode }) => {
    const [language, setLanguageState] = useState<Language>('en');

    useEffect(() => {
        const savedLang = localStorage.getItem('language') as Language;
        if (savedLang) {
            setLanguageState(savedLang);
        }
    }, []);

    const setLanguage = (lang: Language) => {
        setLanguageState(lang);
        localStorage.setItem('language', lang);
    };

    const value = {
        language,
        t: translations[language],
        setLanguage,
    };

    return (
        <LanguageContext.Provider value={value}>
            {children}
        </LanguageContext.Provider>
    );
};

export const useLanguage = () => {
    const context = useContext(LanguageContext);
    if (context === undefined) {
        throw new Error('useLanguage must be used within a LanguageProvider');
    }
    return context;
};
