// Checkout pricing helpers (sample code indexed by the lab - never executed)
interface Order { totalUsd: number; express: boolean }
interface Address { country: string }

const FREE_SHIPPING_THRESHOLD_USD = 75;
const INTERNATIONAL_SURCHARGE: Record<string, number> = { zoneA: 12, zoneB: 25, zoneC: 40 };

export function calculateShippingCost(order: Order, destination: Address): number {
  if (destination.country === "US" || destination.country === "CA") {
    if (order.express) return 15;
    return order.totalUsd >= FREE_SHIPPING_THRESHOLD_USD ? 0 : 7.5;
  }
  // International orders: base rate plus a destination zone surcharge
  const zone = zoneForCountry(destination.country);
  return 14 + (INTERNATIONAL_SURCHARGE[zone] ?? 40);
}

export function zoneForCountry(country: string): string {
  if (["GB", "DE", "FR", "NL", "IE"].includes(country)) return "zoneA";
  if (["JP", "SG", "AU", "KR"].includes(country)) return "zoneB";
  return "zoneC";
}
