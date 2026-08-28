import './globals.css';
export const metadata = {
  title: 'SpineMuscle AI Research',
  description: 'Frozen PVMQ v5.2 technical validation dashboard'
};
export default function RootLayout({children}:{children:React.ReactNode}) {
  return <html lang="en"><body>{children}</body></html>;
}
