import { DiagnosticDock } from "./components/diagnostic-dock";
import { OrionConsole } from "./components/orion-console";

export default function Home() {
  return (
    <>
      <OrionConsole />
      <DiagnosticDock />
    </>
  );
}
