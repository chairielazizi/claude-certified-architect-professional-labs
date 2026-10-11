---
name: facilities-booking
description: Book, list or cancel meeting rooms and desks. Use when the user wants a room for a meeting, a desk for a day, or to cancel a booking.
tools: facilities_book_room, facilities_list_rooms, facilities_cancel_booking
docs: DOC-13
---
# Facilities booking

## When to use
Any request to reserve a meeting room or desk, see available rooms, or cancel a reservation.

## Steps
1. Extract building, number of attendees, start time and duration (default 60 minutes).
2. Resolve relative dates ("tomorrow") against today's date before calling tools.
3. Call `facilities_book_room`; the system picks a room with capacity between 1x and 2x the attendees.
4. If no room is free, call `facilities_list_rooms` and offer the two closest alternatives.
5. Booking rules (no-show release, customer-only floors) are in DOC-13 - load it only if the user asks.

## Output
Room name, capacity, time and booking ID. Remind the user to check in within 10 minutes.
